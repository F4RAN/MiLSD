# eval_postproc.py — post-processing ABLATION BATTERY for the F-Clip line detector (training-free).
# Caches forward passes once, then tries many decode/TTA/filter ideas and reports sAP {5,10,15} for each,
# with the delta vs the trained baseline. Most won't help — the point is to measure, not guess.
# Usage:  python eval_postproc.py [npz] [IN] [WIDTH] [ckpt]
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
print('loaded', CKPT, '| IN', IN, 'WIDTH', WIDTH, 'val', Xva.shape, '| device', DEV)

# ---------- forward cache (raw logits; sin2θ-aware un-flip; multi-scale) ----------
def fwd(x):  # -> (4, h, w) numpy
    with torch.no_grad(): return m(x)[0].cpu().numpy()
def tens(img, S): return torch.from_numpy(cv2.resize(img, (S, S)).astype(np.float32)/255.)[None, None].to(DEV)
def uh(p): p = p[:, :, ::-1].copy(); p[3] = -p[3]; return p     # undo h-flip (negate sin2θ)
def uv(p): p = p[:, ::-1, :].copy(); p[3] = -p[3]; return p     # undo v-flip
t0 = time.time()
F0, FH, FV, FHV, S224, S288 = [], [], [], [], [], []
for i in range(len(Xva)):
    x = tens(Xva[i], IN)
    F0.append(fwd(x)); FH.append(uh(fwd(torch.flip(x, [3])))); FV.append(uv(fwd(torch.flip(x, [2]))))
    FHV.append(uv(uh(fwd(torch.flip(x, [2, 3])))))
    S224.append(fwd(tens(Xva[i], 224))); S288.append(fwd(tens(Xva[i], 288)))
print('cached %d images x6 forwards in %.0fs' % (len(Xva), time.time()-t0))

def compose(i, mode):
    if mode == 'single': return F0[i]
    if mode == 'tta2':   return (F0[i]+FH[i]+FV[i])/3.0
    if mode == 'tta4':   return (F0[i]+FH[i]+FV[i]+FHV[i])/4.0
    if mode == 'tta2p':  # average center in PROB space, geometry raw
        c = (1/(1+np.exp(-F0[i][0]))+1/(1+np.exp(-FH[i][0]))+1/(1+np.exp(-FV[i][0])))/3.0
        g = (F0[i]+FH[i]+FV[i])/3.0; g = g.copy(); g[0] = np.log(np.clip(c, 1e-6, 1-1e-6)/(1-np.clip(c, 1e-6, 1-1e-6))); return g
    raise ValueError(mode)

# ---------- decode (returns segments in the heatmap's own OUT grid) ----------
def decode(p, thr=0.06, peak='parab', nms_k=3, blur=0.0, gwin=1):
    out = p.shape[1]; C = 1/(1+np.exp(-p[0]))
    if blur > 0: C = cv2.GaussianBlur(C, (0, 0), blur)
    LEN, COS, SIN = p[1], p[2], p[3]
    pd = nms_k//2; Pp = np.pad(C, pd); mx = np.zeros_like(C)
    for a in range(nms_k):
        for b in range(nms_k): mx = np.maximum(mx, Pp[a:a+out, b:b+out])
    segs = []
    for iy, ix in np.argwhere((C >= mx) & (C > thr)):
        fx, fy = float(ix), float(iy)
        if peak == 'parab' and 0 < ix < out-1 and 0 < iy < out-1:
            dn = C[iy, ix+1]-C[iy, ix-1]; dd = 2*C[iy, ix]-C[iy, ix+1]-C[iy, ix-1]
            en = C[iy+1, ix]-C[iy-1, ix]; ed = 2*C[iy, ix]-C[iy+1, ix]-C[iy-1, ix]
            if abs(dd) > 1e-6: fx += np.clip(0.5*dn/dd, -1, 1)
            if abs(ed) > 1e-6: fy += np.clip(0.5*en/ed, -1, 1)
        elif peak == 'centroid' and 0 < ix < out-1 and 0 < iy < out-1:
            w = C[iy-1:iy+2, ix-1:ix+2]; sw = w.sum()+1e-9
            ys, xs = np.mgrid[-1:2, -1:2]; fx += (w*xs).sum()/sw; fy += (w*ys).sum()/sw
        if gwin > 1:
            y0, y1 = max(0, iy-1), min(out, iy+2); x0, x1 = max(0, ix-1), min(out, ix+2)
            Lv = LEN[y0:y1, x0:x1].mean(); co = COS[y0:y1, x0:x1].mean(); si = SIN[y0:y1, x0:x1].mean()
        else:
            Lv, co, si = LEN[iy, ix], COS[iy, ix], SIN[iy, ix]
        Lh = Lv*out/2; th = 0.5*np.arctan2(si, co); hx, hy = Lh*np.cos(th), Lh*np.sin(th)
        segs.append([fx-hx, fy-hy, fx+hx, fy+hy, float(C[iy, ix])])
    return segs, out

def to128(segs, out):
    sc = SRC/out
    return [[s[0]*sc, s[1]*sc, s[2]*sc, s[3]*sc, s[4]] for s in segs]

# ---------- post-decode filters (operate in 128-space) ----------
def seg_nms(segs, dthr=2.0):
    segs = sorted(segs, key=lambda s: -s[4]); keep = []
    for s in segs:
        dup = False
        for k in keep:
            d1 = (s[0]-k[0])**2+(s[1]-k[1])**2+(s[2]-k[2])**2+(s[3]-k[3])**2
            d2 = (s[0]-k[2])**2+(s[1]-k[3])**2+(s[2]-k[0])**2+(s[3]-k[1])**2
            if min(d1, d2) < dthr: dup = True; break
        if not dup: keep.append(s)
    return keep
def lfilt(segs, mn): return [s for s in segs if np.hypot(s[2]-s[0], s[3]-s[1]) >= mn]
def rerank(segs, mode):
    o = []
    for x1, y1, x2, y2, s in segs:
        sc = s*np.hypot(x2-x1, y2-y1) if mode == 'len' else (s*s if mode == 'sq' else s)
        o.append([x1, y1, x2, y2, sc])
    return o

# ---------- sAP ----------
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
def sap(segs_per_img):
    res = {5: [], 10: [], 15: []}; ng = 0
    for i, segs in enumerate(segs_per_img):
        if segs: P = np.array([[[s[0], s[1]], [s[2], s[3]]] for s in segs]); S = np.array([s[4] for s in segs])
        else: P = np.zeros((0, 2, 2)); S = np.zeros(0)
        G = np.asarray(Lva[i]).reshape(-1, 2, 2).astype(float); ng += len(G)
        for t in (5, 10, 15): res[t] += match(P, S, G, t)
    return {t: round(100*AP(res[t], ng), 1) for t in (5, 10, 15)}

# ---------- run one config across all images ----------
def run(compose_mode='tta2', dec=None, scales=None, post=None):
    dec = dec or {}; post = post or {}
    per = []
    for i in range(len(Xva)):
        if scales:                                   # multi-scale: pool decoded segments
            segs = []
            for s in scales:
                p = compose(i, compose_mode) if s == IN else (S224[i] if s == 224 else S288[i])
                d, o = decode(p, **dec); segs += to128(d, o)
        else:
            d, o = decode(compose(i, compose_mode), **dec); segs = to128(d, o)
        if post.get('seg_nms'): segs = seg_nms(segs, post['seg_nms'])
        if post.get('lfilt'):   segs = lfilt(segs, post['lfilt'])
        if post.get('rerank'):  segs = rerank(segs, post['rerank'])
        per.append(segs)
    return sap(per)

# ---------- the battery ----------
base = run('single', dict(peak='none'))
print('\n%-34s %s' % ('baseline (trained, raw peaks)', base))
def show(name, r):
    delta = r[10] - base[10]
    print('%-34s %s  (%+.1f)' % (name, r, delta))

tests = [
    ('sub-pixel parabolic',        lambda: run('single', dict(peak='parab'))),
    ('sub-pixel centroid',         lambda: run('single', dict(peak='centroid'))),
    ('TTA2 (h+v flip)',            lambda: run('tta2',   dict(peak='parab'))),
    ('TTA4 (h+v+hv flip)',         lambda: run('tta4',   dict(peak='parab'))),
    ('TTA2 prob-space avg',        lambda: run('tta2p',  dict(peak='parab'))),
    ('TTA2 + centroid',            lambda: run('tta2',   dict(peak='centroid'))),
    ('TTA2 + NMS 5x5',             lambda: run('tta2',   dict(peak='parab', nms_k=5))),
    ('TTA2 + NMS 1x1 (none)',      lambda: run('tta2',   dict(peak='parab', nms_k=1))),
    ('TTA2 + heatmap blur 0.8',    lambda: run('tta2',   dict(peak='parab', blur=0.8))),
    ('TTA2 + geometry-window 3x3', lambda: run('tta2',   dict(peak='parab', gwin=3))),
    ('TTA2 + segment-NMS',         lambda: run('tta2',   dict(peak='parab'), post=dict(seg_nms=2.0))),
    ('TTA2 + length filter (>=3)', lambda: run('tta2',   dict(peak='parab'), post=dict(lfilt=3.0))),
    ('TTA2 + rerank score*len',    lambda: run('tta2',   dict(peak='parab'), post=dict(rerank='len'))),
    ('TTA2 + rerank score^2',      lambda: run('tta2',   dict(peak='parab'), post=dict(rerank='sq'))),
    ('multi-scale 224+256+288',    lambda: run('tta2',   dict(peak='parab'), scales=[224, IN, 288])),
    ('multi-scale + segment-NMS',  lambda: run('tta2',   dict(peak='parab'), scales=[224, IN, 288], post=dict(seg_nms=2.0))),
    ('KITCHEN SINK',               lambda: run('tta4',   dict(peak='centroid', gwin=3), scales=[224, IN, 288], post=dict(seg_nms=2.0))),
]
print('\n%-34s %s' % ('config', '{sAP5, sAP10, sAP15}   (Δ sAP10 vs baseline)'))
print('-'*78)
results = {}
for name, fn in tests:
    t = time.time(); r = fn(); results[name] = r; show(name, r)
best = max(results.items(), key=lambda kv: kv[1][10])
print('-'*78)
print('BEST: %s  ->  %s' % (best[0], best[1]))
