# eval_heuristics2.py — photo-driven heuristics for the faint/wrong lines, measured (sAP + operating-point P/R).
# Three signals chosen from looking at the kitchen panel:
#   (1) ORIENTATION ALIGNMENT (LSD-style): real edge => image gradient is PERPENDICULAR to the line along it.
#   (2) ENDPOINT SUPPORT: over-extended lines have endpoints floating off any edge (mean-support misses this).
#   (3) ANGLE-AWARE LINE-NMS: merge near-collinear overlapping duplicates (exact-endpoint NMS missed them).
# Usage:  python eval_heuristics2.py [npz] [IN] [WIDTH] [ckpt] [gate]
import os, sys, time, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

DEV = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
NPZ = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else os.path.expanduser('~/Desktop/Projects/LSD-TML/implementation/wireframe_lines_256_5000.npz')
IN  = int(sys.argv[2]) if len(sys.argv) > 2 else 256
WIDTH = int(sys.argv[3]) if len(sys.argv) > 3 else 4
CKPT = sys.argv[4] if len(sys.argv) > 4 else 'fclip_%d_best.pt' % IN
GATE = float(sys.argv[5]) if len(sys.argv) > 5 else 0.25
OUT = IN // 2; SRC = 128; REDUCE = max(8, 8 * WIDTH)
ARCH = [(8*WIDTH, 2), (16*WIDTH, 2), (32*WIDTH, 1), (32*WIDTH, 1), (32*WIDTH, 1)]

d = np.load(NPZ, allow_pickle=True); Xva, Lva = d['Xva'], d['Lva']
class FCNet(nn.Module):
    def __init__(s, oc=4):
        super().__init__(); L = []; c = 1
        for co, st in ARCH: L += [nn.Conv2d(c, co, 3, st, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True)]; c = co
        s.backbone = nn.Sequential(*L); s.reduce = nn.Conv2d(c, REDUCE, 1); s.head = nn.Conv2d(REDUCE, oc, 3, 1, 1)
    def forward(s, x): return s.head(F.interpolate(s.reduce(s.backbone(x)), scale_factor=2, mode='nearest'))
m = FCNet().to(DEV); m.load_state_dict(torch.load(CKPT, map_location=DEV)); m.eval()
print('loaded', CKPT, '| val', Xva.shape)

def fwd(x):
    with torch.no_grad(): return m(x)[0].cpu().numpy()
def uh(p): p = p[:, :, ::-1].copy(); p[3] = -p[3]; return p
def uv(p): p = p[:, ::-1, :].copy(); p[3] = -p[3]; return p
def decode(p, thr=0.06):
    out = p.shape[1]; C = 1/(1+np.exp(-p[0])); LEN, COS, SIN = p[1], p[2], p[3]
    Pp = np.pad(C, 1); mx = np.zeros_like(C)
    for a in range(3):
        for b in range(3): mx = np.maximum(mx, Pp[a:a+out, b:b+out])
    segs = []
    for iy, ix in np.argwhere((C >= mx) & (C > thr)):
        fx, fy = float(ix), float(iy)
        if 0 < ix < out-1 and 0 < iy < out-1:
            dn = C[iy, ix+1]-C[iy, ix-1]; dd = 2*C[iy, ix]-C[iy, ix+1]-C[iy, ix-1]
            en = C[iy+1, ix]-C[iy-1, ix]; ed = 2*C[iy, ix]-C[iy+1, ix]-C[iy-1, ix]
            if abs(dd) > 1e-6: fx += np.clip(0.5*dn/dd, -1, 1)
            if abs(ed) > 1e-6: fy += np.clip(0.5*en/ed, -1, 1)
        Lh = LEN[iy, ix]*out/2; th = 0.5*np.arctan2(SIN[iy, ix], COS[iy, ix]); hx, hy = Lh*np.cos(th), Lh*np.sin(th)
        segs.append([fx-hx, fy-hy, fx+hx, fy+hy, float(C[iy, ix])])
    return [[s[0]*SRC/out, s[1]*SRC/out, s[2]*SRC/out, s[3]*SRC/out, s[4]] for s in segs]

# cache TTA4 decode + gradient components (gx, gy, mag) in 128-space
t0 = time.time(); BASE = []; GX = []; GY = []; MAG = []
for i in range(len(Xva)):
    x = torch.from_numpy(cv2.resize(Xva[i], (IN, IN)).astype(np.float32)/255.)[None, None].to(DEV)
    p = (fwd(x) + uh(fwd(torch.flip(x, [3]))) + uv(fwd(torch.flip(x, [2]))) + uv(uh(fwd(torch.flip(x, [2, 3]))))) / 4.0
    BASE.append(decode(p))
    g = cv2.GaussianBlur(cv2.resize(Xva[i], (SRC, SRC)).astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    mg = np.abs(gx)+np.abs(gy); GX.append(gx); GY.append(gy); MAG.append(mg/(mg.max()+1e-6))
print('cached %d in %.0fs' % (len(Xva), time.time()-t0))

def samp(seg, n=20):
    xs = np.clip(np.linspace(seg[0], seg[2], n), 0, SRC-1).astype(int)
    ys = np.clip(np.linspace(seg[1], seg[3], n), 0, SRC-1).astype(int)
    return xs, ys
def align(seg, gx, gy, mag, tol=22.5, magthr=0.06):       # (1) fraction of points where grad ⟂ line
    xs, ys = samp(seg); la = np.arctan2(seg[3]-seg[1], seg[2]-seg[0])
    ga = np.arctan2(gy[ys, xs], gx[ys, xs])
    perp_err = np.abs(((ga - la) % np.pi) - np.pi/2)        # 0 when perpendicular
    ok = (perp_err < np.radians(tol)) & (mag[ys, xs] > magthr)
    return ok.mean()
def end_support(seg, mag, k=3):                            # (2) min gradient at the two endpoints
    def at(x, y):
        x, y = int(np.clip(x, 0, SRC-1)), int(np.clip(y, 0, SRC-1))
        return mag[max(0, y-k):y+k+1, max(0, x-k):x+k+1].max()
    return min(at(seg[0], seg[1]), at(seg[2], seg[3]))

# heuristics
def H_identity(segs, i): return segs
def H_align_filter(thr):
    return lambda segs, i: [s for s in segs if align(s, GX[i], GY[i], MAG[i]) >= thr]
def H_align_rescore(segs, i): return [[*s[:4], s[4]*(0.3+0.7*align(s, GX[i], GY[i], MAG[i]))] for s in segs]
def H_endsupport(thr):
    return lambda segs, i: [s for s in segs if end_support(s, MAG[i]) >= thr]
def line_nms(segs, ang_tol=8, dist_tol=2.5):
    segs = sorted(segs, key=lambda s: -s[4]); keep = []
    for s in segs:
        a = np.degrees(np.arctan2(s[3]-s[1], s[2]-s[0])) % 180; mx, my = (s[0]+s[2])/2, (s[1]+s[3])/2; dup = False
        for k in keep:
            ak = np.degrees(np.arctan2(k[3]-k[1], k[2]-k[0])) % 180
            if min(abs(a-ak), 180-abs(a-ak)) < ang_tol:
                vx, vy = k[2]-k[0], k[3]-k[1]; L2 = vx*vx+vy*vy+1e-9
                t = max(0, min(1, ((mx-k[0])*vx+(my-k[1])*vy)/L2)); px, py = k[0]+t*vx, k[1]+t*vy
                if (mx-px)**2+(my-py)**2 < dist_tol*dist_tol: dup = True; break
        if not dup: keep.append(s)
    return keep
def H_linenms(segs, i): return line_nms(segs)
def chain(*hs):
    def f(segs, i):
        for h in hs: segs = h(segs, i)
        return segs
    return f

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
def evaluate(H):
    res = {5: [], 10: [], 15: []}; ng = 0; TP = FP = 0; nseg = 0
    for i in range(len(Xva)):
        segs = H(BASE[i], i)
        if segs: P = np.array([[[s[0], s[1]], [s[2], s[3]]] for s in segs]); S = np.array([s[4] for s in segs])
        else: P = np.zeros((0, 2, 2)); S = np.zeros(0)
        G = np.asarray(Lva[i]).reshape(-1, 2, 2).astype(float); ng += len(G)
        for t in (5, 10, 15): res[t] += match(P, S, G, t)
        gp = [s for s in segs if s[4] >= GATE]; nseg += len(gp)
        if gp:
            Pg = np.array([[[s[0], s[1]], [s[2], s[3]]] for s in gp]); Sg = np.array([s[4] for s in gp])
            fl = match(Pg, Sg, G, 10.0); tp = sum(f for _, f in fl); TP += tp; FP += len(fl)-tp
    sap = {t: round(100*AP(res[t], ng), 1) for t in (5, 10, 15)}
    prec = TP/max(TP+FP, 1); rec = TP/max(ng, 1); f1 = 2*prec*rec/max(prec+rec, 1e-9)
    return sap, prec, rec, f1, nseg/len(Xva)

tests = [
    ('baseline (TTA4)',                H_identity),
    ('orientation-align filter >=0.30', H_align_filter(0.30)),
    ('orientation-align filter >=0.50', H_align_filter(0.50)),
    ('orientation-align re-score',      H_align_rescore),
    ('endpoint-support >=0.15',         H_endsupport(0.15)),
    ('endpoint-support >=0.30',         H_endsupport(0.30)),
    ('angle-aware line-NMS',            H_linenms),
    ('line-NMS + align>=0.30',          chain(H_linenms, H_align_filter(0.30))),
    ('line-NMS + endpoint>=0.15',       chain(H_linenms, H_endsupport(0.15))),
    ('ALL (nms+align.3+endpt.15)',      chain(H_linenms, H_align_filter(0.30), H_endsupport(0.15))),
]
print('\n%-36s %-26s %s' % ('heuristic', 'sAP {5,10,15}', 'op@%.2f: prec / rec / F1 / lines' % GATE))
print('-'*104)
b = None
for name, H in tests:
    sap, pr, rc, f1, ln = evaluate(H)
    if b is None: b = (sap[10], f1)
    print('%-36s %-26s %.3f / %.3f / %.3f / %.0f   (sAP %+.1f, F1 %+.3f)'
          % (name, str(sap), pr, rc, f1, ln, sap[10]-b[0], f1-b[1]))
print('-'*104)
print('Goal here is OPERATING-POINT precision/F1 (cleaner output), not sAP (settled at 21.0).')
