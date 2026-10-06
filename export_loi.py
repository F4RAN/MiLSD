# export_loi.py — export the trained LoI MLP to an int8 QDQ ONNX file (148 -> 256 -> 128 -> 1).
# Calibration inputs are real LoI features from TRAIN images, computed with the int8 backbone.
# Usage:  python export_loi.py [loi.pt] [backbone_int8.onnx] [npz] [out.onnx]
import sys, numpy as np, torch, onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, CalibrationMethod
from milsd import LoI, LOI_DIN, prep, decode, loi_features

LOI = sys.argv[1] if len(sys.argv) > 1 else 'loi_256.pt'
BB = sys.argv[2] if len(sys.argv) > 2 else 'milsd_int8.onnx'
NPZ = sys.argv[3] if len(sys.argv) > 3 else 'data/wireframe_lines_256_5000.npz'
OUT = sys.argv[4] if len(sys.argv) > 4 else 'loi_int8.onnx'

loi = LoI(); loi.load_state_dict(torch.load(LOI, map_location='cpu')); loi.eval()
fp = OUT.replace('.onnx', '_fp32.onnx')
torch.onnx.export(loi.net, torch.zeros(1, LOI_DIN), fp, input_names=['x'], output_names=['logit'],
                  dynamic_axes={'x': {0: 'n'}, 'logit': {0: 'n'}}, opset_version=13, dynamo=False)
so = ort.InferenceSession(BB, providers=['CPUExecutionProvider'])
Xtr = np.load(NPZ, allow_pickle=True)['Xtr']; batches = []
for i in np.linspace(0, len(Xtr)-1, 100).astype(int):
    c, l, a, f = so.run(None, {'input': prep(Xtr[i]).numpy()}); maps = np.concatenate([c, l, a], 1)[0]
    lines, sc = decode(maps)
    if len(lines): batches.append(loi_features(torch.from_numpy(f), maps, lines, sc).numpy())
class Reader(CalibrationDataReader):
    def __init__(s): s.it = iter(batches)
    def get_next(s):
        b = next(s.it, None); return None if b is None else {'x': b}
quantize_static(fp, OUT, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax)
print('wrote', OUT, '| fp32 reference', fp)
