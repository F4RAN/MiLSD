# eval_threshold.py — what a HARD score floor does to the numbers.
# Reported sAP uses ALL detections (threshold-free). This shows the *deployed* numbers if you discard below T:
# floored-sAP (area under the truncated PR curve) + operating-point precision/recall/F1 + lines/img.
# Usage:  python eval_threshold.py [npz] [IN] [WIDTH] [ckpt]
import os, sys, time, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

DEV = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
NPZ = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else os.path.expanduser('~/Desktop/Projects/LSD-TML/implementation/wireframe_lines_256_5000.npz')
IN  = int(sys.argv[2]) if len(sys.argv) > 2 else 256
WIDTH = int(sys.argv[3]) if len(sys.argv) > 3 else 4
CKPT = sys.argv[4] if len(sys.argv) > 4 else 'fclip_%d_best.pt' % IN
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

t0 = time.time(); BASE = []
for i in range(len(Xva)):
    x = torch.from_numpy(cv2.resize(Xva[i], (IN, IN)).astype(np.float32)/255.)[None, None].to(DEV)
    p = (fwd(x) + uh(fwd(torch.flip(x, [3]))) + uv(fwd(torch.flip(x, [2]))) + uv(uh(fwd(torch.flip(x, [2, 3]))))) / 4.0
    BASE.append(decode(p))
print('cached %d (TTA4) in %.0fs | raw avg %.0f lines/img' % (len(Xva), time.time()-t0, np.mean([len(b) for b in BASE])))

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
def evalT(T):
    res = {5: [], 10: [], 15: []}; ng = 0; TP = FP = 0; nseg = 0
    for i in range(len(Xva)):
        segs = [s for s in BASE[i] if s[4] >= T]; nseg += len(segs)
        if segs: P = np.array([[[s[0], s[1]], [s[2], s[3]]] for s in segs]); S = np.array([s[4] for s in segs])
        else: P = np.zeros((0, 2, 2)); S = np.zeros(0)
        G = np.asarray(Lva[i]).reshape(-1, 2, 2).astype(float); ng += len(G)
        for t in (5, 10, 15): res[t] += match(P, S, G, t)
        fl = match(P, S, G, 10.0); tp = sum(f for _, f in fl); TP += tp; FP += len(fl)-tp
    sap = {t: round(100*AP(res[t], ng), 1) for t in (5, 10, 15)}
    prec = TP/max(TP+FP, 1); rec = TP/max(ng, 1); f1 = 2*prec*rec/max(prec+rec, 1e-9)
    return sap, prec, rec, f1, nseg/len(Xva)

print('\n%-12s %-26s %s' % ('score floor', 'sAP {5,10,15} (floored)', 'prec / rec / F1 / lines'))
print('-'*80)
base10 = None
for T in (0.06, 0.30, 0.40, 0.50, 0.60, 0.70):
    sap, pr, rc, f1, ln = evalT(T)
    if base10 is None: base10 = sap[10]
    tag = '  <- reported sAP (all detections)' if T == 0.06 else '  (sAP10 %+.1f vs reported)' % (sap[10]-base10)
    print('>= %.2f      %-26s %.3f / %.3f / %.3f / %.0f%s' % (T, str(sap), pr, rc, f1, ln, tag))
print('-'*80)
print('Row 0.06 = your REPORTED sAP (uses all detections). Higher floors = a deployed operating point:')
print('precision UP, recall DOWN, and floored-sAP DOWN (you discarded real lines). The headline 21.0 is unchanged.')
