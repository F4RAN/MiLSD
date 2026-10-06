# viz_kitchen.py — render the BEST model (W4 + TTA4 + sub-pixel) on the kitchen val image.
# Produces kitchen_best.png:  input | ground truth | ours.
# Usage:  python viz_kitchen.py [npz] [IN] [WIDTH] [ckpt] [idx] [viz_thr]
import os, sys, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

DEV = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
NPZ = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else 'data/wireframe_lines_256_5000.npz'
IN  = int(sys.argv[2]) if len(sys.argv) > 2 else 256
WIDTH = int(sys.argv[3]) if len(sys.argv) > 3 else 4
CKPT = sys.argv[4] if len(sys.argv) > 4 else 'fclip_%d_best.pt' % IN
IDX = int(sys.argv[5]) if len(sys.argv) > 5 else 40           # the kitchen example
VTHR = float(sys.argv[6]) if len(sys.argv) > 6 else 0.25
OUT = IN // 2; SRC = 128; REDUCE = max(8, 8 * WIDTH); DISP = 512
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
        segs.append((fx-hx, fy-hy, fx+hx, fy+hy, float(C[iy, ix])))
    return segs, out

img = Xva[IDX]
x = torch.from_numpy(cv2.resize(img, (IN, IN)).astype(np.float32)/255.)[None, None].to(DEV)
p = (fwd(x) + uh(fwd(torch.flip(x, [3]))) + uv(fwd(torch.flip(x, [2]))) + uv(uh(fwd(torch.flip(x, [2, 3]))))) / 4.0
segs, out = decode(p)

def label(im, txt):
    cv2.rectangle(im, (0, 0), (DISP, 28), (0, 0, 0), -1)
    cv2.putText(im, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA); return im
base = cv2.cvtColor(cv2.resize(img, (DISP, DISP)), cv2.COLOR_GRAY2BGR)
gt = np.asarray(Lva[IDX]).reshape(-1, 4) * (DISP/SRC)
gimg = base.copy()
for x1, y1, x2, y2 in gt: cv2.line(gimg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 230, 0), 1, cv2.LINE_AA)
pimg = base.copy(); scp = DISP/out
cand = sorted([s for s in segs if s[4] >= VTHR], key=lambda t: -t[4])
for x1, y1, x2, y2, s in reversed(cand):                                # draw faint first, bright on top
    c = int(80 + 175*min(1, (s-VTHR)/(1-VTHR)))
    cv2.line(pimg, (int(x1*scp), int(y1*scp)), (int(x2*scp), int(y2*scp)), (0, c, 0), 1, cv2.LINE_AA)
cap = 'ours W4+TTA4: %d lines (score>%.2f)' % (len(cand), VTHR)
panel = np.concatenate([label(base.copy(), 'input (val #%d)' % IDX),
                        label(gimg, 'ground truth: %d lines' % len(gt)),
                        label(pimg, cap)], 1)
cv2.imwrite('kitchen_best.png', panel)
print('saved kitchen_best.png | GT %d lines | drew %d / %d total (score>%.2f)' % (len(gt), len(cand), len(segs), VTHR))
