# eval_pretrained_ref.py — REFERENCE TEST: run the OFFICIAL pretrained M-LSD-tiny (the ~56-at-512 model)
# on YOUR val set, at several input sizes, with its OWN 4-channel inference path. This tells us the true
# resolution ceiling and validates our eval harness — separating "my estimate was wrong" from "training is wrong".
#
# Run:  python eval_pretrained_ref.py
import os, sys, json, numpy as np, cv2, torch
import torch.nn.functional as F

REPO = os.environ.get('MLSD_REPO', 'mlsd_pytorch')
os.chdir(REPO); sys.path.insert(0, REPO)
from models.mbv2_mlsd_tiny import MobileV2_MLSD_Tiny   # 4-ch inference model that matches the shipped checkpoint

DEV = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'models/mlsd_tiny_512_fp32.pth'
VAL  = 'data/wireframe_raw/valid.json'
IMG  = 'data/wireframe_raw/images/'

def deccode(tpMap, topk_n=500, ksize=3):
    b, c, h, w = tpMap.shape
    disp = tpMap[:, 1:5, :, :][0]
    heat = torch.sigmoid(tpMap[:, 0, :, :])
    hmax = F.max_pool2d(heat, (ksize, ksize), stride=1, padding=(ksize-1)//2)
    heat = (heat * (hmax == heat).float()).reshape(-1)
    scores, idx = torch.topk(heat, topk_n, dim=-1, largest=True)
    yy = torch.div(idx, w, rounding_mode='floor').unsqueeze(-1)
    xx = torch.fmod(idx, w).unsqueeze(-1)
    ptss = torch.cat((yy, xx), dim=-1).cpu().numpy()
    return ptss, scores.cpu().numpy(), disp.cpu().numpy().transpose((1, 2, 0))

def infer(image, model, S, score_thr=0.05, dist_thr=5.0):
    h, w, _ = image.shape; hr, wr = h/S, w/S
    ri = np.concatenate([cv2.resize(image, (S, S), interpolation=cv2.INTER_AREA),
                         np.ones([S, S, 1])], axis=-1)                      # <-- 4th channel = ones (official)
    bi = ((ri.transpose(2, 0, 1)[None].astype('float32')) / 127.5) - 1.0    # <-- official normalization
    with torch.no_grad():
        out = model(torch.from_numpy(bi).float().to(DEV))
    pts, sc, vmap = deccode(out, 500, 3)
    dmap = np.sqrt(((vmap[:, :, :2] - vmap[:, :, 2:]) ** 2).sum(-1))
    lines, scores = [], []
    for (y, x), s in zip(pts, sc):
        if s > score_thr and dmap[y, x] > dist_thr:
            dxs, dys, dxe, dye = vmap[y, x, :]
            lines.append([x+dxs, y+dys, x+dxe, y+dye]); scores.append(float(s))
    if not lines:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    L = 2 * np.array(lines, np.float32)
    L[:, 0] *= wr; L[:, 1] *= hr; L[:, 2] *= wr; L[:, 3] *= hr
    return L, np.array(scores, np.float32)

def msTPFP(line_pred, line_gt, thr=10.0):
    lp = line_pred.reshape(-1, 2, 2)[:, :, ::-1]; lg = line_gt.reshape(-1, 2, 2)[:, :, ::-1]
    diff = ((lp[:, None, :, None] - lg[:, None]) ** 2).sum(-1)
    diff = np.minimum(diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0])
    choice = np.argmin(diff, 1); dist = np.min(diff, 1)
    hit = np.zeros(len(lg), bool); tp = np.zeros(len(lp)); fp = np.zeros(len(lp))
    for i in range(len(lp)):
        if dist[i] < thr and not hit[choice[i]]:
            hit[choice[i]] = True; tp[i] = 1
        else: fp[i] = 1
    return tp, fp

def AP(tp, fp):
    rec = np.concatenate(([0.0], tp, [1.0])); pre = np.concatenate(([0.0], tp/np.maximum(tp+fp, 1e-9), [0.0]))
    for i in range(pre.size-1, 0, -1): pre[i-1] = max(pre[i-1], pre[i])
    i = np.where(rec[1:] != rec[:-1])[0]
    return np.sum((rec[i+1]-rec[i]) * pre[i+1])

def evaluate(S):
    model = MobileV2_MLSD_Tiny().to(DEV).eval()
    model.load_state_dict(torch.load(CKPT, map_location=DEV), strict=True)
    gts = json.load(open(VAL)); tp_l, fp_l, sc_l = [], [], []; n_gt = 0
    for c in gts:
        img = cv2.cvtColor(cv2.imread(IMG + c['filename']), cv2.COLOR_BGR2RGB)
        pl, ps = infer(img, model, S)
        s = np.array([128.0/c['width'], 128.0/c['height'], 128.0/c['width'], 128.0/c['height']], np.float32)
        gl = np.array(c['lines'], np.float32).reshape(-1, 4) * s
        n_gt += len(gl)
        if len(pl) == 0: continue
        tp, fp = msTPFP(pl*s, gl, 10.0) if len(gl) else (np.zeros(len(pl)), np.ones(len(pl)))
        tp_l.append(tp); fp_l.append(fp); sc_l.append(ps)
    tp = np.concatenate(tp_l); fp = np.concatenate(fp_l); sc = np.concatenate(sc_l)
    idx = np.argsort(sc)[::-1]
    return AP(np.cumsum(tp[idx])/n_gt, np.cumsum(fp[idx])/n_gt) * 100

if __name__ == '__main__':
    print('device:', DEV, '| val images:', len(json.load(open(VAL))))
    for S in [512, 256, 160]:
        print('PRETRAINED M-LSD-tiny @ %3d  ->  sAP10 = %.2f' % (S, evaluate(S)))
