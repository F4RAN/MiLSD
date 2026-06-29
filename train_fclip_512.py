# train_fclip_512.py — F-Clip (center+length+angle) at HIGH RESOLUTION on M1, with sAP eval built in.
# Tests your "resolution is the dominant lever" thesis on your best model, at a size only F-Clip can fit.
# Usage:  python train_fclip_512.py [npz_path] [IN_SIZE]      (defaults: 5000-img npz, 512)
#   e.g.  python ~/Desktop/Projects/LSD-TML/local_m1/train_fclip_512.py            # 512
#         python ~/Desktop/Projects/LSD-TML/local_m1/train_fclip_512.py "" 384     # 384
import os, sys, numpy as np, cv2, random, torch, torch.nn as nn, torch.nn.functional as F

SEED = 0; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

NPZ = (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else
       'data/wireframe_lines_256_5000.npz')
IN_SIZE  = int(sys.argv[2]) if len(sys.argv) > 2 else 256        # 256 = native (real pixels); >256 upscales (fake res)
WIDTH    = int(sys.argv[3]) if len(sys.argv) > 3 else 1          # capacity lever: channel multiplier (1=25k params)
OUT_SIZE = IN_SIZE // 2; SRC = 128
REDUCE   = max(8, 8 * WIDTH)
EPOCHS   = int(sys.argv[4]) if len(sys.argv) > 4 else 150       # 4th arg: epochs (use 300 for the final)
BATCH, LR = 16, 1e-3
ARCH = [(8*WIDTH, 2), (16*WIDTH, 2), (32*WIDTH, 1), (32*WIDTH, 1), (32*WIDTH, 1)]   # WIDTH scales capacity

d = np.load(NPZ, allow_pickle=True)
Xtr, Ltr, Xva, Lva = d['Xtr'], d['Ltr'], d['Xva'], d['Lva']
print('device', DEV, '| npz', os.path.basename(NPZ), '| train', Xtr.shape, 'val', Xva.shape,
      '| IN', IN_SIZE, 'OUT', OUT_SIZE, 'WIDTH', WIDTH, 'REDUCE', REDUCE)
if IN_SIZE > int(Xtr.shape[1]):
    print('NOTE: IN_SIZE > native image size (%d) -> UPSCALING (fake resolution). Capacity (WIDTH) is the real lever here.' % int(Xtr.shape[1]))

def make_target(lines, out=OUT_SIZE, src=SRC, r=2):
    C = np.zeros((out, out), np.float32); G = np.zeros((3, out, out), np.float32); s = out / src
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        X1, Y1, X2, Y2 = x1*s, y1*s, x2*s, y2*s; cx, cy = (X1+X2)/2, (Y1+Y2)/2
        ix, iy = int(round(cx)), int(round(cy))
        if not (0 <= ix < out and 0 <= iy < out): continue
        Ln = np.hypot(X2-X1, Y2-Y1)/out; th = np.arctan2(Y2-Y1, X2-X1)
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                xx, yy = ix+dx, iy+dy
                if 0 <= xx < out and 0 <= yy < out:
                    C[yy, xx] = max(C[yy, xx], np.exp(-(dx*dx+dy*dy)/(2*(r/2.0)**2)))
                    G[0, yy, xx] = Ln; G[1, yy, xx] = np.cos(2*th); G[2, yy, xx] = np.sin(2*th)
    return C, G

class DS(torch.utils.data.Dataset):   # targets built on the fly -> 512 doesn't blow up RAM
    def __init__(self, X, L, aug): self.X, self.L, self.aug = X, L, aug
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = cv2.resize(self.X[i], (IN_SIZE, IN_SIZE)).astype(np.float32) / 255.
        C, G = make_target(self.L[i])
        if self.aug and random.random() < 0.5: x = x[:, ::-1]; C = C[:, ::-1]; G = G[:, :, ::-1].copy(); G[2] = -G[2]
        if self.aug and random.random() < 0.5: x = x[::-1, :]; C = C[::-1, :]; G = G[:, ::-1, :].copy(); G[2] = -G[2]
        return (torch.from_numpy(np.ascontiguousarray(x))[None],
                torch.from_numpy(np.ascontiguousarray(C)), torch.from_numpy(np.ascontiguousarray(G)))

class FCNet(nn.Module):
    def __init__(self, out_ch=4):
        super().__init__(); Lr = []; c = 1
        for co, s in ARCH: Lr += [nn.Conv2d(c, co, 3, s, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]; c = co
        self.backbone = nn.Sequential(*Lr); self.reduce = nn.Conv2d(c, REDUCE, 1); self.head = nn.Conv2d(REDUCE, out_ch, 3, 1, 1)
    def forward(self, x):
        return self.head(F.interpolate(self.reduce(self.backbone(x)), scale_factor=2, mode='nearest'))

model = FCNet().to(DEV)
wp = sum(p.numel() for p in model.parameters() if p.dim() > 1)
peak = {'a': 0}
def hook(m, i, o): peak['a'] = max(peak['a'], sum(t.numel() for t in i if torch.is_tensor(t)) + o.numel())
hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
with torch.no_grad(): model(torch.zeros(1, 1, IN_SIZE, IN_SIZE).to(DEV))
[h.remove() for h in hs]
print('params %d (~%.0f KB int8) | peak conv arena ~%.0f KB int8 -> H7 1MB: %s'
      % (wp, wp/1024, peak['a']/1024, 'FITS' if peak['a']/1024 < 1024 else 'OVER (trim channels later)'))

samp = np.stack([make_target(Ltr[i])[0] for i in range(min(200, len(Ltr)))])
pf = float((samp >= 0.999).mean()); POSW = torch.tensor([min((1-pf)/max(pf, 1e-6), 50.0)], device=DEV)
tl = torch.utils.data.DataLoader(DS(Xtr, Ltr, True), batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)  # macOS spawn -> 0 workers
def loss_fn(pred, C, G):
    lc = F.binary_cross_entropy_with_logits(pred[:, 0], C, pos_weight=POSW)
    m = (C > 0.1).float(); lg = (F.l1_loss(pred[:, 1:4], G, reduction='none').sum(1)*m).sum()/(m.sum()+1e-6)
    return lc + 2.0*lg
opt = torch.optim.Adam(model.parameters(), LR); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

Xvf = np.stack([cv2.resize(im, (IN_SIZE, IN_SIZE)) for im in Xva]).astype(np.float32)/255.
def decode(p, out=OUT_SIZE, thr=0.1):
    C = 1/(1+np.exp(-p[0])); LEN, COS, SIN = p[1], p[2], p[3]; Pp = np.pad(C, 1); mx = np.zeros_like(C)
    for a in range(3):
        for b in range(3): mx = np.maximum(mx, Pp[a:a+out, b:b+out])
    segs = []
    for iy, ix in np.argwhere((C >= mx) & (C > thr)):
        Lh = LEN[iy, ix]*out/2; th = 0.5*np.arctan2(SIN[iy, ix], COS[iy, ix]); hx, hy = Lh*np.cos(th), Lh*np.sin(th)
        segs.append((ix-hx, iy-hy, ix+hx, iy+hy, float(C[iy, ix])))
    return segs
def match2(P, S, Gt, thr):
    order = np.argsort(-S); matched = np.zeros(len(Gt), bool); fl = []
    for idx in order:
        if len(Gt) == 0: fl.append((S[idx], 0)); continue
        p = P[idx]
        d1 = ((p[0]-Gt[:, 0])**2).sum(1)+((p[1]-Gt[:, 1])**2).sum(1); d2 = ((p[0]-Gt[:, 1])**2).sum(1)+((p[1]-Gt[:, 0])**2).sum(1)
        d = np.minimum(d1, d2); j = int(np.argmin(d))
        if d[j] <= thr and not matched[j]: matched[j] = True; fl.append((S[idx], 1))
        else: fl.append((S[idx], 0))
    return fl
def AP(fl, ng):
    fl = sorted(fl, key=lambda t: -t[0]); tp = np.cumsum([f for _, f in fl]); fp = np.cumsum([1-f for _, f in fl])
    rec = tp/max(ng, 1); prec = tp/np.maximum(tp+fp, 1e-9); mrec = np.r_[0, rec, 1]; mpre = np.r_[0, prec, 0]
    for i in range(len(mpre)-1, 0, -1): mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]; return float(np.sum((mrec[idx+1]-mrec[idx])*mpre[idx+1]))
def eval_sap(m):
    m.eval(); res = {5: [], 10: [], 15: []}; ng = 0; sc = SRC/OUT_SIZE
    with torch.no_grad():
        for i in range(len(Xva)):
            pr = m(torch.from_numpy(Xvf[i])[None, None].to(DEV))[0].cpu().numpy(); segs = decode(pr)
            if segs:
                P = np.array([[[x1*sc, y1*sc], [x2*sc, y2*sc]] for x1, y1, x2, y2, _ in segs]); S = np.array([s for *_, s in segs])
            else: P = np.zeros((0, 2, 2)); S = np.zeros(0)
            G = np.asarray(Lva[i]).reshape(-1, 2, 2).astype(float); ng += len(G)
            for thr in (5, 10, 15): res[thr] += match2(P, S, G, thr)
    return {t: round(100*AP(res[t], ng), 1) for t in (5, 10, 15)}

best = 0.0
for ep in range(EPOCHS):
    model.train(); run = 0; nb = 0
    for x, C, G in tl:
        x, C, G = x.to(DEV), C.to(DEV), G.to(DEV)
        loss = loss_fn(model(x), C, G); opt.zero_grad(); loss.backward(); opt.step(); run += loss.item(); nb += 1
    sch.step()
    if ep % 15 == 0 or ep == EPOCHS-1:
        sap = eval_sap(model)
        if sap[10] > best: best = sap[10]; torch.save(model.state_dict(), 'fclip_%d_best.pt' % IN_SIZE)
        print('epoch %3d  train %.3f  sAP %s  (best sAP10 %.1f)' % (ep, run/max(nb, 1), sap, best))
print('DONE @ %d. best sAP10 = %.1f  (your 256px F-Clip best was 10.6)' % (IN_SIZE, best))
