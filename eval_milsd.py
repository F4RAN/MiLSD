# eval_milsd.py — sAP of the full MiLSD pipeline on Wireframe-val, in fp32 or with the int8 ONNX files.
# Usage:  python eval_milsd.py [--int8] [--tta] [--loi] [--npz PATH]
#   fp32:  python eval_milsd.py --loi            (backbone fclip_256_best.pt, LoI loi_256.pt)
#   int8:  python eval_milsd.py --int8 --loi     (milsd_int8.onnx + loi_int8.onnx, as deployed)
import argparse, time, numpy as np, torch
from milsd import load_model, prep, tta_maps, decode, loi_features, LoI, sap

ap = argparse.ArgumentParser()
ap.add_argument('--int8', action='store_true'); ap.add_argument('--tta', action='store_true')
ap.add_argument('--loi', action='store_true'); ap.add_argument('--npz', default='data/wireframe_lines_256_5000.npz')
ap.add_argument('--ckpt', default='fclip_256_best.pt'); ap.add_argument('--onnx', default='milsd_int8.onnx')
ap.add_argument('--loi_pt', default='loi_256.pt'); ap.add_argument('--loi_onnx', default='loi_int8.onnx')
A = ap.parse_args()
d = np.load(A.npz, allow_pickle=True); Xva, Lva = d['Xva'], d['Lva']

if A.int8:
    import onnxruntime as ort
    so = ort.InferenceSession(A.onnx, providers=['CPUExecutionProvider'])
    def run(x):  # -> maps (4,OUT,OUT), feat (1,32,64,64)
        c, l, a, f = so.run(None, {'input': x.numpy()}); return np.concatenate([c, l, a], 1)[0], torch.from_numpy(f)
    if A.loi:
        sl = ort.InferenceSession(A.loi_onnx, providers=['CPUExecutionProvider'])
        verify = lambda X: 1/(1+np.exp(-sl.run(None, {'x': X.numpy()})[0].reshape(-1)))
else:
    m = load_model(A.ckpt)
    def run(x):
        with torch.no_grad(): out, f = m.featfwd(x)
        return out[0].numpy(), f
    if A.loi:
        loi = LoI(); loi.load_state_dict(torch.load(A.loi_pt, map_location='cpu')); loi.eval()
        def verify(X):
            with torch.no_grad(): return torch.sigmoid(loi(X)).numpy()

t0 = time.time(); preds, ncand = [], []
for img in Xva:
    x = prep(img); maps, feat = run(x)
    heat = tta_maps(lambda t: run(t)[0], x) if A.tta else maps
    lines, sc = decode(heat); ncand.append(len(lines))
    if A.loi and len(lines):
        sc = verify(loi_features(feat, maps, lines, sc)) * sc     # LoI reads the un-flipped pass
    preds.append((lines, sc))
print('%s%s%s  sAP %s  | candidates/img mean %.0f p95 %.0f max %d | %.0fs' % (
      'int8' if A.int8 else 'fp32', ' +TTA4' if A.tta else '', ' +LoI' if A.loi else '',
      sap(preds, Lva), np.mean(ncand), np.percentile(ncand, 95), np.max(ncand), time.time()-t0))
