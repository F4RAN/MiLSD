# milsd.py — single source of truth for the MiLSD model, decoder, LoI verifier and sAP metric.
# The other scripts import from here instead of keeping their own copies.
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

IN, WIDTH = 256, 4
OUT = IN // 2; SRC = 128; REDUCE = max(8, 8*WIDTH)
ARCH = [(8*WIDTH, 2), (16*WIDTH, 2), (32*WIDTH, 1), (32*WIDTH, 1), (32*WIDTH, 1)]
THR = 0.06                       # center-confidence threshold for candidates
NPTS, MXP = 32, 4                # LoI: points sampled along each line, max-pooled down to MXP bins


# ---------------------------------------------------------------- model
class FCNet(nn.Module):
    """F-Clip backbone. Output: 4 maps at OUT x OUT = (center logit, length, cos2θ, sin2θ)."""
    def __init__(s, oc=4):
        super().__init__(); L = []; c = 1
        for co, st in ARCH: L += [nn.Conv2d(c, co, 3, st, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True)]; c = co
        s.backbone = nn.Sequential(*L); s.reduce = nn.Conv2d(c, REDUCE, 1); s.head = nn.Conv2d(REDUCE, oc, 3, 1, 1)
    def forward(s, x): return s.head(F.interpolate(s.reduce(s.backbone(x)), scale_factor=2, mode='nearest'))
    def featfwd(s, x):
        """returns (4-map output, 32-ch reduce feature at 64x64) — the feature the LoI head reads."""
        r = s.reduce(s.backbone(x)); return s.head(F.interpolate(r, scale_factor=2, mode='nearest')), r


class SplitHead(nn.Module):
    """Deployment form of FCNet: the head is split into 3 convs so each output gets its own int8 scale.
    Outputs: center (1 ch), length (1 ch), angle (2 ch), feat (32-ch reduce map for the LoI head).
    Numerically identical to FCNet in fp32."""
    def __init__(s, m):
        super().__init__(); s.backbone, s.reduce = m.backbone, m.reduce
        W, b = m.head.weight.data, m.head.bias.data
        s.h_center = nn.Conv2d(REDUCE, 1, 3, 1, 1); s.h_length = nn.Conv2d(REDUCE, 1, 3, 1, 1); s.h_angle = nn.Conv2d(REDUCE, 2, 3, 1, 1)
        for conv, sl in ((s.h_center, slice(0, 1)), (s.h_length, slice(1, 2)), (s.h_angle, slice(2, 4))):
            conv.weight.data, conv.bias.data = W[sl].clone(), b[sl].clone()
    def forward(s, x):
        r = s.reduce(s.backbone(x)); f = F.interpolate(r, scale_factor=2, mode='nearest')
        return s.h_center(f), s.h_length(f), s.h_angle(f), r


def load_model(ckpt='fclip_256_best.pt'):
    m = FCNet(); m.load_state_dict(torch.load(ckpt, map_location='cpu')); m.eval()
    for p in m.parameters(): p.requires_grad = False
    return m


def prep(img):
    """uint8 grayscale image -> (1,1,IN,IN) float tensor in [0,1]"""
    return torch.from_numpy(cv2.resize(img, (IN, IN)).astype(np.float32)/255.)[None, None]


# ---------------------------------------------------------------- TTA
def unflip_h(p): p = p[:, :, ::-1].copy(); p[3] = -p[3]; return p   # undo h-flip (negate sin2θ)
def unflip_v(p): p = p[:, ::-1, :].copy(); p[3] = -p[3]; return p   # undo v-flip


def tta_maps(fwd, x):
    """fwd: (1,1,IN,IN) tensor -> (4,OUT,OUT) numpy. Averages the image and its h / v / hv flips."""
    return (fwd(x) + unflip_h(fwd(torch.flip(x, [3]))) + unflip_v(fwd(torch.flip(x, [2])))
            + unflip_v(unflip_h(fwd(torch.flip(x, [2, 3]))))) / 4.0


# ---------------------------------------------------------------- decode
def decode(p, thr=THR):
    """(4,OUT,OUT) maps -> lines (M,4) in 128-px space, scores (M,).
    3x3 peak NMS with a tie-safe rule (strict '>' against the 4 earlier neighbours, '>=' against the 4 later
    ones), so a flat top of equal values gives exactly ONE peak. With int8 outputs, equal neighbours are
    common; the old 'C >= max(3x3)' rule kept all of them and reported the same line several times.
    Then 1-D parabolic sub-pixel refinement and endpoint reconstruction."""
    C = 1/(1+np.exp(-p[0])); LEN, COS, SIN = p[1], p[2], p[3]
    Pp = np.pad(C, 1, constant_values=-1.0); keep = C > thr
    for a in range(3):
        for b in range(3):
            if (a, b) == (1, 1): continue
            nb = Pp[a:a+OUT, b:b+OUT]
            keep &= (C > nb) if (a, b) < (1, 1) else (C >= nb)
    L, S = [], []
    for iy, ix in np.argwhere(keep):
        fx, fy = float(ix), float(iy)
        if 0 < ix < OUT-1 and 0 < iy < OUT-1:
            dn = C[iy, ix+1]-C[iy, ix-1]; dd = 2*C[iy, ix]-C[iy, ix+1]-C[iy, ix-1]
            en = C[iy+1, ix]-C[iy-1, ix]; ed = 2*C[iy, ix]-C[iy+1, ix]-C[iy-1, ix]
            if abs(dd) > 1e-6: fx += np.clip(0.5*dn/dd, -1, 1)
            if abs(ed) > 1e-6: fy += np.clip(0.5*en/ed, -1, 1)
        Lh = LEN[iy, ix]*OUT/2; th = 0.5*np.arctan2(SIN[iy, ix], COS[iy, ix]); hx, hy = Lh*np.cos(th), Lh*np.sin(th)
        L.append([fx-hx, fy-hy, fx+hx, fy+hy]); S.append(float(C[iy, ix]))
    sc = SRC/OUT
    return np.array(L, np.float32).reshape(-1, 4)*sc, np.array(S, np.float32)


# ---------------------------------------------------------------- LoI verifier
def loi_pool(feat, lines):
    """feat (1,C,H,W); lines (M,4) in 128-px space -> (M, C*MXP).
    NPTS bilinear samples along each line, max-pooled down to MXP bins."""
    t = torch.linspace(0, 1, NPTS).view(1, NPTS)
    x = lines[:, 0:1]*(1-t) + lines[:, 2:3]*t; y = lines[:, 1:2]*(1-t) + lines[:, 3:4]*t
    g = torch.stack([x/(SRC-1)*2-1, y/(SRC-1)*2-1], dim=-1).unsqueeze(0)
    s = F.grid_sample(feat, g, align_corners=True)                       # (1,C,M,NPTS)
    s = F.max_pool1d(s[0].permute(1, 0, 2), kernel_size=NPTS//MXP)       # (M,C,MXP)
    return s.reshape(lines.shape[0], -1)


def loi_features(feat, maps, lines, scores):
    """LoI input for each candidate: pooled 32-ch reduce feature, pooled output maps, geometry.
    The center map enters as a probability (0..1), not a logit, so all inputs share a small range —
    this keeps a single int8 input scale usable on the MCU."""
    lt = torch.as_tensor(lines, dtype=torch.float32)
    mp = torch.as_tensor(np.ascontiguousarray(maps), dtype=torch.float32)[None].clone()
    mp[:, 0] = torch.sigmoid(mp[:, 0])
    dx = lines[:, 2]-lines[:, 0]; dy = lines[:, 3]-lines[:, 1]; th = np.arctan2(dy, dx)
    geo = np.stack([np.hypot(dx, dy)/SRC, np.cos(2*th), np.sin(2*th), scores], 1).astype(np.float32)
    return torch.cat([loi_pool(feat, lt), loi_pool(mp, lt), torch.from_numpy(geo)], 1)


LOI_DIN = REDUCE*MXP + 4*MXP + 4      # 148


class LoI(nn.Module):
    def __init__(s, din=LOI_DIN):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(din, 256), nn.ReLU(True), nn.Dropout(0.3),
                              nn.Linear(256, 128), nn.ReLU(True), nn.Linear(128, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


def match_labels(lines, scores, G, thr=10):
    """greedy one-to-one match against ground truth (same rule as sAP10) -> 0/1 label per candidate"""
    order = np.argsort(-scores); mt = np.zeros(len(G), bool); lab = np.zeros(len(lines), np.float32); Gf = G.reshape(-1, 4)
    for idx in order:
        if len(Gf) == 0: break
        p = lines[idx]
        d1 = (p[0]-Gf[:, 0])**2+(p[1]-Gf[:, 1])**2+(p[2]-Gf[:, 2])**2+(p[3]-Gf[:, 3])**2
        d2 = (p[0]-Gf[:, 2])**2+(p[1]-Gf[:, 3])**2+(p[2]-Gf[:, 0])**2+(p[3]-Gf[:, 1])**2
        dd = np.minimum(d1, d2); j = int(np.argmin(dd))
        if dd[j] <= thr and not mt[j]: mt[j] = True; lab[idx] = 1.0
    return lab


# ---------------------------------------------------------------- sAP
def _match(P, S, G, thr):
    o = np.argsort(-S, kind='stable'); mt = np.zeros(len(G), bool); fl = []
    for idx in o:
        if len(G) == 0: fl.append((S[idx], 0)); continue
        p = P[idx]; d1 = ((p[0]-G[:, 0])**2).sum(1)+((p[1]-G[:, 1])**2).sum(1); d2 = ((p[0]-G[:, 1])**2).sum(1)+((p[1]-G[:, 0])**2).sum(1)
        dd = np.minimum(d1, d2); j = int(np.argmin(dd))
        if dd[j] <= thr and not mt[j]: mt[j] = True; fl.append((S[idx], 1))
        else: fl.append((S[idx], 0))
    return fl


def _ap(fl, ng):
    fl = sorted(fl, key=lambda t: -t[0]); tp = np.cumsum([f for _, f in fl]); fp = np.cumsum([1-f for _, f in fl])
    rec = tp/max(ng, 1); pr = tp/np.maximum(tp+fp, 1e-9); mr = np.r_[0, rec, 1]; mp = np.r_[0, pr, 0]
    for i in range(len(mp)-1, 0, -1): mp[i-1] = max(mp[i-1], mp[i])
    ix = np.where(mr[1:] != mr[:-1])[0]; return float(np.sum((mr[ix+1]-mr[ix])*mp[ix+1]))


def sap(preds, gts):
    """preds: list of (lines (M,4), scores (M,)) in 128-px space; gts: list of GT line arrays."""
    r = {5: [], 10: [], 15: []}; ng = 0
    for (L, S), G in zip(preds, gts):
        G = np.asarray(G).reshape(-1, 2, 2).astype(float); ng += len(G)
        for t in (5, 10, 15): r[t] += _match(L.reshape(-1, 2, 2), S, G, t)
    return {t: round(100*_ap(r[t], ng), 1) for t in (5, 10, 15)}
