# train_loi.py — L-CNN/HAWP-style LoI (Line-of-Interest) VERIFICATION HEAD on the FROZEN F-Clip backbone.
# Pools features along each candidate line, classifies real/fake, and re-scores lines by that verification
# score. Trains ONLY the small head (backbone frozen). Checks if it lifts sAP past the 17.8 (single)/21.0 (TTA).
# Usage:  python train_loi.py [npz] [IN] [WIDTH] [ckpt]
import os, sys, time, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

DEV = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
NPZ = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else 'data/wireframe_lines_256_5000.npz'
IN  = int(sys.argv[2]) if len(sys.argv) > 2 else 256
WIDTH = int(sys.argv[3]) if len(sys.argv) > 3 else 4
CKPT = sys.argv[4] if len(sys.argv) > 4 else 'fclip_%d_best.pt' % IN
OUT = IN // 2; SRC = 128; REDUCE = max(8, 8 * WIDTH)
ARCH = [(8*WIDTH, 2), (16*WIDTH, 2), (32*WIDTH, 1), (32*WIDTH, 1), (32*WIDTH, 1)]
NPTS = 32; MXP = 4; NTRAIN = 3000; TOPC = 150; HEPOCHS = 80   # richer: 128-ch backbone feat, 32 pts max-pooled to 4, 2x data

d = np.load(NPZ, allow_pickle=True); Xtr, Ltr, Xva, Lva = d['Xtr'], d['Ltr'], d['Xva'], d['Lva']
class FCNet(nn.Module):
    def __init__(s, oc=4):
        super().__init__(); L = []; c = 1
        for co, st in ARCH: L += [nn.Conv2d(c, co, 3, st, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True)]; c = co
        s.backbone = nn.Sequential(*L); s.reduce = nn.Conv2d(c, REDUCE, 1); s.head = nn.Conv2d(REDUCE, oc, 3, 1, 1)
    def forward(s, x): return s.head(F.interpolate(s.reduce(s.backbone(x)), scale_factor=2, mode='nearest'))
    def featfwd(s, x):                                  # returns (4-ch output, 128-ch BACKBONE feature, richer)
        b = s.backbone(x); f = F.interpolate(s.reduce(b), scale_factor=2, mode='nearest'); return s.head(f), b
m = FCNet().to(DEV); m.load_state_dict(torch.load(CKPT, map_location=DEV)); m.eval()
for p in m.parameters(): p.requires_grad = False
print('loaded', CKPT, '| backbone FROZEN | feat ch =', REDUCE, '| device', DEV)

def tens(img): return torch.from_numpy(cv2.resize(img, (IN, IN)).astype(np.float32)/255.)[None, None].to(DEV)
def decode(out):                                        # out (4,OUT,OUT) numpy -> lines(M,4) grid, scores(M,)
    C = 1/(1+np.exp(-out[0])); LEN, COS, SIN = out[1], out[2], out[3]
    Pp = np.pad(C, 1); mx = np.zeros_like(C)
    for a in range(3):
        for b in range(3): mx = np.maximum(mx, Pp[a:a+OUT, b:b+OUT])
    L, S = [], []
    for iy, ix in np.argwhere((C >= mx) & (C > 0.06)):
        fx, fy = float(ix), float(iy)
        if 0 < ix < OUT-1 and 0 < iy < OUT-1:
            dn = C[iy, ix+1]-C[iy, ix-1]; dd = 2*C[iy, ix]-C[iy, ix+1]-C[iy, ix-1]
            en = C[iy+1, ix]-C[iy-1, ix]; ed = 2*C[iy, ix]-C[iy+1, ix]-C[iy-1, ix]
            if abs(dd) > 1e-6: fx += np.clip(0.5*dn/dd, -1, 1)
            if abs(ed) > 1e-6: fy += np.clip(0.5*en/ed, -1, 1)
        Lh = LEN[iy, ix]*OUT/2; th = 0.5*np.arctan2(SIN[iy, ix], COS[iy, ix]); hx, hy = Lh*np.cos(th), Lh*np.sin(th)
        L.append([fx-hx, fy-hy, fx+hx, fy+hy]); S.append(float(C[iy, ix]))
    return np.array(L, np.float32).reshape(-1, 4), np.array(S, np.float32)

def loi_pool(feat, lines):                              # NPTS pts along each line, max-pooled to MXP -> (M, C*MXP)
    t = torch.linspace(0, 1, NPTS, device=feat.device).view(1, NPTS)
    x = lines[:, 0:1]*(1-t) + lines[:, 2:3]*t; y = lines[:, 1:2]*(1-t) + lines[:, 3:4]*t
    g = torch.stack([x/(SRC-1)*2-1, y/(SRC-1)*2-1], dim=-1).unsqueeze(0)     # normalize by line space (SRC), not feat res
    s = F.grid_sample(feat, g, align_corners=True)                          # (1,C,M,NPTS)
    s = F.max_pool1d(s[0].permute(1, 0, 2), kernel_size=NPTS//MXP)          # (M,C,MXP)
    return s.reshape(lines.shape[0], -1)                                    # (M, C*MXP)
def geom(lines, scores):                                # (M,4)+scores -> (M,4): len, cos2θ, sin2θ, center-score
    dx = lines[:, 2]-lines[:, 0]; dy = lines[:, 3]-lines[:, 1]; Ln = np.hypot(dx, dy); th = np.arctan2(dy, dx)
    return np.stack([Ln/SRC, np.cos(2*th), np.sin(2*th), scores], 1).astype(np.float32)
def featurize(b, out_t, lines, sc):                     # lines: numpy (M,4); backbone + head-output along line + geom
    lt = torch.tensor(lines, dtype=torch.float32, device=DEV)
    return torch.cat([loi_pool(b, lt), loi_pool(out_t, lt), torch.tensor(geom(lines, sc), device=DEV)], 1)
def label(lines, scores, G):                            # greedy one-to-one match (= the sAP definition)
    order = np.argsort(-scores); mt = np.zeros(len(G), bool); lab = np.zeros(len(lines), np.float32); Gf = G.reshape(-1, 4)
    for idx in order:
        p = lines[idx]
        d1 = (p[0]-Gf[:, 0])**2+(p[1]-Gf[:, 1])**2+(p[2]-Gf[:, 2])**2+(p[3]-Gf[:, 3])**2
        d2 = (p[0]-Gf[:, 2])**2+(p[1]-Gf[:, 3])**2+(p[2]-Gf[:, 0])**2+(p[3]-Gf[:, 1])**2
        dd = np.minimum(d1, d2); j = int(np.argmin(dd))
        if dd[j] <= 10 and not mt[j]: mt[j] = True; lab[idx] = 1.0
    return lab

class LoI(nn.Module):
    def __init__(s, din):
        super().__init__(); s.net = nn.Sequential(nn.Linear(din, 256), nn.ReLU(True), nn.Dropout(0.3),
                                                  nn.Linear(256, 128), nn.ReLU(True), nn.Linear(128, 1))
    def forward(s, x): return s.net(x).squeeze(-1)

# ---- build training cache (frozen-backbone features are fixed -> cache once) ----
t0 = time.time(); FE, LA = [], []
for i in range(min(NTRAIN, len(Xtr))):
    with torch.no_grad(): out, b = m.featfwd(tens(Xtr[i]))
    lines, sc = decode(out[0].cpu().numpy())
    if len(lines) == 0: continue
    o = np.argsort(-sc)[:TOPC]; lines, sc = lines[o], sc[o]
    G = np.asarray(Ltr[i]).reshape(-1, 2, 2).astype(np.float32)
    lab = label(lines, sc, G)
    with torch.no_grad(): feat = featurize(b, out, lines, sc)
    FE.append(feat.cpu()); LA.append(torch.tensor(lab))
FE = torch.cat(FE).to(DEV); LA = torch.cat(LA).to(DEV)
print('cache: %d candidates (%.1f%% positive) in %.0fs | feat dim %d' % (len(FE), 100*LA.mean().item(), time.time()-t0, FE.shape[1]))

# ---- train ONLY the LoI head ----
loi = LoI(FE.shape[1]).to(DEV); opt = torch.optim.Adam(loi.parameters(), 1e-3, weight_decay=1e-4)
pw = torch.tensor([(LA == 0).sum()/max((LA == 1).sum(), 1)], device=DEV)
for ep in range(HEPOCHS):
    loi.train(); perm = torch.randperm(len(FE), device=DEV)
    for k in range(0, len(perm), 8192):
        idx = perm[k:k+8192]; loss = F.binary_cross_entropy_with_logits(loi(FE[idx]), LA[idx], pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
loi.eval(); print('head trained (%d epochs)' % HEPOCHS)

# ---- eval: rank val candidates by center-score (baseline) vs LoI vs LoI*center ----
def match(P, S, G, thr):
    o = np.argsort(-S); mt = np.zeros(len(G), bool); fl = []
    for idx in o:
        if len(G) == 0: fl.append((S[idx], 0)); continue
        p = P[idx]; d1 = ((p[0]-G[:, 0])**2).sum(1)+((p[1]-G[:, 1])**2).sum(1); d2 = ((p[0]-G[:, 1])**2).sum(1)+((p[1]-G[:, 0])**2).sum(1)
        dd = np.minimum(d1, d2); j = int(np.argmin(dd))
        if dd[j] <= thr and not mt[j]: mt[j] = True; fl.append((S[idx], 1))
        else: fl.append((S[idx], 0))
    return fl
def AP(fl, ng):
    fl = sorted(fl, key=lambda t: -t[0]); tp = np.cumsum([f for _, f in fl]); fp = np.cumsum([1-f for _, f in fl])
    rec = tp/max(ng, 1); pr = tp/np.maximum(tp+fp, 1e-9); mr = np.r_[0, rec, 1]; mp = np.r_[0, pr, 0]
    for i in range(len(mp)-1, 0, -1): mp[i-1] = max(mp[i-1], mp[i])
    ix = np.where(mr[1:] != mr[:-1])[0]; return float(np.sum((mr[ix+1]-mr[ix])*mp[ix+1]))
def fwd(x):
    with torch.no_grad(): return m(x)[0].cpu().numpy()
def uh(p): p = p[:, :, ::-1].copy(); p[3] = -p[3]; return p
def uv(p): p = p[:, ::-1, :].copy(); p[3] = -p[3]; return p
def eval_all(tta=False):
    cache = []
    for i in range(len(Xva)):
        x = tens(Xva[i])
        with torch.no_grad(): out_s, b = m.featfwd(x)
        heat = (fwd(x)+uh(fwd(torch.flip(x, [3])))+uv(fwd(torch.flip(x, [2])))+uv(uh(fwd(torch.flip(x, [2, 3])))))/4.0 if tta else out_s[0].cpu().numpy()
        lines, sc = decode(heat); G = np.asarray(Lva[i]).reshape(-1, 2, 2).astype(float)
        if len(lines) == 0: cache.append((np.zeros((0, 2, 2)), np.zeros(0), np.zeros(0), np.zeros(0), G)); continue
        with torch.no_grad():
            ls = torch.sigmoid(loi(featurize(b, out_s, lines, sc))).cpu().numpy()
        lab = label(lines, sc, G)                              # true TP/FP labels -> oracle (perfect-verifier) ceiling
        cache.append((lines.reshape(-1, 2, 2), sc, ls, lab, G))
    res = {}
    for rank in ('center', 'loi', 'loixc', 'oracle'):
        r = {5: [], 10: [], 15: []}; ng = 0
        for P, sc, ls, lab, G in cache:
            S = {'center': sc, 'loi': ls, 'loixc': ls*sc, 'oracle': lab*1e3+sc}[rank]; ng += len(G)
            for t in (5, 10, 15): r[t] += match(P, S, G, t)
        res[rank] = {t: round(100*AP(r[t], ng), 1) for t in (5, 10, 15)}
    return res
S1 = eval_all(tta=False); S4 = eval_all(tta=True)
print('\n                          sAP {5, 10, 15}')
print('single  center (baseline) ', S1['center'])
print('single  LoI               ', S1['loi'])
print('single  LoI x center      ', S1['loixc'])
print('TTA4    center            ', S4['center'])
print('TTA4    LoI                ', S4['loi'])
print('TTA4    LoI x center  <<<  ', S4['loixc'])
print('TTA4    ORACLE (ceiling)   ', S4['oracle'], '  <- perfect verifier on these candidates = recall ceiling')

# ---- example figures: GT (green) | ours = LoI-verified + dedup (yellow), several val images ----
import math
def same_edge(a, b, ang_tol=15, perp_tol=3.0, min_ov=0.2):   # two segments = same physical edge?
    ax, ay, bx, by = b[0], b[1], b[2], b[3]; vx, vy = bx-ax, by-ay; Lb = math.hypot(vx, vy)+1e-9; ux, uy = vx/Lb, vy/Lb
    aa = math.atan2(a[3]-a[1], a[2]-a[0]); ab = math.atan2(vy, vx)
    if math.degrees(abs(((aa-ab) % math.pi)-math.pi/2)) < 90-ang_tol: return False
    def proj(px, py): t = (px-ax)*ux+(py-ay)*uy; return math.hypot(px-(ax+t*ux), py-(ay+t*uy)), t
    d1, t1 = proj(a[0], a[1]); d2, t2 = proj(a[2], a[3])
    if max(d1, d2) > perp_tol: return False
    lo, hi = min(t1, t2), max(t1, t2); ov = max(0.0, min(hi, Lb)-max(lo, 0.0))
    return ov/(min(hi-lo, Lb)+1e-9) >= min_ov
DISP = 460; SCD = DISP/SRC; VTHRK = 0.4; IDXS = [40, 12, 170, 300]
def lab(im, t):
    cv2.rectangle(im, (0, 0), (DISP, 26), (0, 0, 0), -1); cv2.putText(im, t, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA); return im
def render(idx):
    x = tens(Xva[idx])
    with torch.no_grad(): outx, bx = m.featfwd(x)
    heat = (fwd(x)+uh(fwd(torch.flip(x, [3])))+uv(fwd(torch.flip(x, [2])))+uv(uh(fwd(torch.flip(x, [2, 3])))))/4.0
    lx, scx = decode(heat)
    with torch.no_grad(): lsx = torch.sigmoid(loi(featurize(bx, outx, lx, scx))).cpu().numpy()
    finx = lsx*scx; fnx = (finx-finx.min())/(finx.max()-finx.min()+1e-9)
    base = cv2.cvtColor(cv2.resize(Xva[idx], (DISP, DISP)), cv2.COLOR_GRAY2BGR)
    gt = np.asarray(Lva[idx]).reshape(-1, 4); g = base.copy()
    for x1, y1, x2, y2 in gt*SCD: cv2.line(g, (int(x1), int(y1)), (int(x2), int(y2)), (0, 230, 0), 1, cv2.LINE_AA)
    cand = sorted([i for i in range(len(lx)) if scx[i] >= VTHRK], key=lambda i: -fnx[i]); kept = []
    for i in cand:
        if not any(same_edge(lx[i], lx[k]) for k in kept): kept.append(i)
    o = base.copy()
    for i in sorted(kept, key=lambda i: fnx[i]):
        x1, y1, x2, y2 = lx[i]*SCD; c = int(120+135*fnx[i]); cv2.line(o, (int(x1), int(y1)), (int(x2), int(y2)), (0, c, c), 2, cv2.LINE_AA)
    return np.concatenate([lab(g, 'GT #%d: %d lines' % (idx, len(gt))), lab(o, 'ours: %d (LoI+dedup)' % len(kept))], 1)
cv2.imwrite('examples.png', np.concatenate([render(i) for i in IDXS], 0))
print('saved examples.png  (%d examples, each row:  GT | ours)' % len(IDXS))
