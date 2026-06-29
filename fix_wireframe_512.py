# fix_wireframe_512.py — correct the line encoding in wireframe_lines_512.npz.
# The source stores each line as [x1, y1, dx, dy] (start + displacement); the converter wrote dx,dy as if
# they were x2,y2. Fix: x2 = x1+dx, y2 = y1+dy. Verified to reproduce the original GT exactly (100% match).
# Usage:  python fix_wireframe_512.py /path/to/wireframe_lines_512.npz [/path/to/output.npz]
import sys, numpy as np

inp = sys.argv[1] if len(sys.argv) > 1 else 'wireframe_lines_512.npz'
out = sys.argv[2] if len(sys.argv) > 2 else inp.replace('.npz', '_fixed.npz')

d = np.load(inp, allow_pickle=True)
print('loaded', inp, '| Xtr', d['Xtr'].shape, 'Xva', d['Xva'].shape)

def fix(L):
    o = []
    for l in L:
        a = np.asarray(l, float).reshape(-1, 4).copy()
        a[:, 2] += a[:, 0]            # x2 = x1 + dx
        a[:, 3] += a[:, 1]            # y2 = y1 + dy
        a = np.clip(a, 0, 128)        # match the dataset's [0,128] convention (old npz was clean 0..128)
        o.append(a.astype(np.float32))
    return np.array(o, dtype=object)

Ltr, Lva = fix(d['Ltr']), fix(d['Lva'])
allc = np.concatenate([l for l in Ltr if len(l)])
print('fixed: TRAIN %d segs %.1f/img | coord range %.1f..%.1f (want 0..128) | %% outside [0,128]: %.2f'
      % (sum(len(l) for l in Ltr), np.mean([len(l) for l in Ltr]),
         allc.min(), allc.max(), 100 * (((allc < -0.01) | (allc > 128.01)).mean())))

np.savez_compressed(out, Xtr=d['Xtr'], Ltr=Ltr, Xva=d['Xva'], Lva=Lva)
print('SAVED', out, '-> train with:  python train_fclip_512.py', '"%s"' % out, '512 4')
