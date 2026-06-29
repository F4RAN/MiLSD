# eval_heuristics.py — heuristic post-filters to remove the faint/wrong lines, MEASURED two ways:
#   (1) sAP {5,10,15}  — ranking quality (only re-scoring heuristics can move this)
#   (2) precision/recall/F1 at a fixed confidence gate — captures the visual "cleanup" (removal helps this)
# Star heuristic = EDGE SUPPORT: a real line lies on an image gradient; faint hallucinations don't.
# Usage:  python eval_heuristics.py [npz] [IN] [WIDTH] [ckpt] [gate]
import os, sys, time, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

DEV = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
NPZ = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else os.path.expanduser('~/Desktop/Projects/LSD-TML/implementation/wireframe_lines_256_5000.npz')
IN  = int(sys.argv[2]) if len(sys.argv) > 2 else 256
WIDTH = int(sys.argv[3]) if len(sys.argv) > 3 else 4
CKPT = sys.argv[4] if len(sys.argv) > 4 else 'fclip_%d_best.pt' % IN
GATE = float(sys.argv[5]) if len(sys.argv) > 5 else 0.25     # operating-point confidence gate for P/R
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
print('loaded', CKPT, '| IN', IN, 'WIDTH', WIDTH, 'val', Xva.shape)

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
        segs.append([fx-hx, fy-hy, fx+hx, fy+hy, float(C[iy, ix])])   # OUT-grid coords
    return [[s[0]*SRC/out, s[1]*SRC/out, s[2]*SRC/out, s[3]*SRC/out, s[4]] for s in segs]   # -> 128-space

# ---- cache TTA4 decode + per-image gradient (edge) maps ----
t0 = time.time(); BASE = []; GRAD = []
for i in range(len(Xva)):
    x = torch.from_numpy(cv2.resize(Xva[i], (IN, IN)).astype(np.float32)/255.)[None, None].to(DEV)
    p = (fwd(x) + uh(fwd(torch.flip(x, [3]))) + uv(fwd(torch.flip(x, [2]))) + uv(uh(fwd(torch.flip(x, [2, 3]))))) / 4.0
    BASE.append(decode(p))
    g = cv2.resize(Xva[i], (SRC, SRC)).astype(np.float32)               # gradient in 128-space (matches line coords)
    gm = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    gm = cv2.GaussianBlur(gm, (0, 0), 1.0)                              # tolerate ~1px endpoint error
    GRAD.append(gm / (gm.max() + 1e-6))
print('cached %d (TTA4 decode + edge maps) in %.0fs' % (len(Xva), time.time()-t0))

def support(seg, grad):                                                # mean edge strength under the segment
    n = 16; xs = np.clip(np.linspace(seg[0], seg[2], n), 0, SRC-1).astype(int)
    ys = np.clip(np.linspace(seg[1], seg[3], n), 0, SRC-1).astype(int)
    return float(grad[ys, xs].mean())

# ---- heuristics: take a per-image segment list, return a modified one ----
def H_identity(segs, g): return segs
def H_scoregate(thr):
    return lambda segs, g: [s for s in segs if s[4] >= thr]
def H_edge_rescore(power=1.0):
    def f(segs, g): return [[*s[:4], s[4]*support(s, g)**power] for s in segs]
    return f
def H_edge_filter(thr):
    def f(segs, g): return [s for s in segs if support(s, g) >= thr]
    return f
def chain(*hs):
    def f(segs, g):
        for h in hs: segs = h(segs, g)
        return segs
    return f

# ---- metrics ----
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
        segs = H(BASE[i], GRAD[i])
        if segs: P = np.array([[[s[0], s[1]], [s[2], s[3]]] for s in segs]); S = np.array([s[4] for s in segs])
        else: P = np.zeros((0, 2, 2)); S = np.zeros(0)
        G = np.asarray(Lva[i]).reshape(-1, 2, 2).astype(float); ng += len(G)
        for t in (5, 10, 15): res[t] += match(P, S, G, t)
        # operating point: gate by CURRENT score, count TP/FP at match-thr 10
        gp = [s for s in segs if s[4] >= GATE]; nseg += len(gp)
        if gp:
            Pg = np.array([[[s[0], s[1]], [s[2], s[3]]] for s in gp]); Sg = np.array([s[4] for s in gp])
            fl = match(Pg, Sg, G, 10.0); tp = sum(f for _, f in fl); TP += tp; FP += len(fl)-tp
    sap = {t: round(100*AP(res[t], ng), 1) for t in (5, 10, 15)}
    prec = TP/max(TP+FP, 1); rec = TP/max(ng, 1); f1 = 2*prec*rec/max(prec+rec, 1e-9)
    return sap, dict(prec=round(prec, 3), rec=round(rec, 3), f1=round(f1, 3), lines=round(nseg/len(Xva), 1))

tests = [
    ('baseline (TTA4, no filter)',     H_identity),
    ('score gate >=0.25',              H_scoregate(0.25)),
    ('score gate >=0.40',              H_scoregate(0.40)),
    ('edge re-score x support',        H_edge_rescore(1.0)),
    ('edge re-score x support^0.5',    H_edge_rescore(0.5)),
    ('edge filter support>=0.10',      H_edge_filter(0.10)),
    ('edge filter support>=0.20',      H_edge_filter(0.20)),
    ('edge re-score + filter>=0.10',   chain(H_edge_filter(0.10), H_edge_rescore(1.0))),
    ('edge filter>=0.15 + score>=0.25', chain(H_edge_filter(0.15), H_scoregate(0.25))),
    ('CLEAN combo (filt.15+resc+gate)', chain(H_edge_filter(0.15), H_edge_rescore(1.0), H_scoregate(0.25))),
]
print('\n%-36s %-26s %s' % ('heuristic', 'sAP {5,10,15}', 'op@%.2f: prec / rec / F1 / lines' % GATE))
print('-'*104)
base_sap = None
for name, H in tests:
    sap, op = evaluate(H)
    if base_sap is None: base_sap = sap[10]
    print('%-36s %-26s %.3f / %.3f / %.3f / %.0f   (sAP10 %+.1f)'
          % (name, str(sap), op['prec'], op['rec'], op['f1'], op['lines'], sap[10]-base_sap))
print('-'*104)
print('Read it: sAP10 only moves with RE-SCORING (ranking). prec/F1 up with FILTERING (the visual cleanup).')
