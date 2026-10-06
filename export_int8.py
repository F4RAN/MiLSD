# export_int8.py — export the MiLSD backbone to an int8 QDQ ONNX file for ST Edge AI / CMSIS-NN.
#
# Fixes the old export, which wrote all 4 output maps (center, length, cos2θ, sin2θ) into ONE int8 tensor
# with ONE scale. The center logits span about [-13, 8], so the int8 step (~0.08) was too coarse for
# length / cos / sin and sAP10 fell from 18.1 to 8.6. Now:
#   * the head is split into 3 convs -> 3 output tensors (center, length, angle), each with its own scale;
#   * weights are quantized per output channel;
#   * a 4th output, 'feat' (the 32-ch reduce map), is exposed for the LoI verifier.
# All of these are plain int8 features. Use the tie-safe decode in milsd.py with this model.
#
# Usage:  python export_int8.py [ckpt] [npz] [out.onnx]
import sys, numpy as np, cv2, torch
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, CalibrationMethod
from onnxruntime.quantization.shape_inference import quant_pre_process
from milsd import load_model, SplitHead, IN

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'fclip_256_best.pt'
NPZ = sys.argv[2] if len(sys.argv) > 2 else 'data/wireframe_lines_256_5000.npz'
OUT = sys.argv[3] if len(sys.argv) > 3 else 'milsd_int8.onnx'
NCAL = 200      # calibration images, taken from the TRAIN split only

m = load_model(CKPT); net = SplitHead(m).eval()
with torch.no_grad():                                   # sanity: split head == original head
    x = torch.rand(1, 1, IN, IN); c, l, a, _ = net(x)
    assert torch.allclose(m(x), torch.cat([c, l, a], 1), atol=1e-5), 'split head mismatch'
fp = OUT.replace('.onnx', '_fp32.onnx'); pre = OUT.replace('.onnx', '_pre.onnx')
torch.onnx.export(net, torch.zeros(1, 1, IN, IN), fp, input_names=['input'],
                  output_names=['center', 'length', 'angle', 'feat'], opset_version=13, dynamo=False)
quant_pre_process(fp, pre)                               # shape inference + BatchNorm folding

Xtr = np.load(NPZ, allow_pickle=True)['Xtr']
cal = [Xtr[i] for i in np.linspace(0, len(Xtr)-1, NCAL).astype(int)]
class Reader(CalibrationDataReader):
    def __init__(s): s.it = iter(cal)
    def get_next(s):
        im = next(s.it, None)
        return None if im is None else {'input': (cv2.resize(im, (IN, IN)).astype(np.float32)/255.)[None, None]}
quantize_static(pre, OUT, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax)
print('wrote', OUT, '| fp32 reference', fp)
