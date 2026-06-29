# convert_local.py — convert your .npz to the repo format (images/*.jpg + train.json/valid.json).
# Usage (from the cloned repo root):  python <thispath> /path/to/wireframe_lines_256_5000.npz
# Schema is locked to the repo loader: list of {filename, lines:[[x1,y1,x2,y2]], height, width}; images MUST be .jpg.
import sys, os, glob, json, numpy as np, cv2

if len(sys.argv) > 1:
    npz = sys.argv[1]
else:
    cand = (glob.glob(os.path.expanduser('~/Downloads/*wireframe*lines*.npz'))
            + glob.glob('*.npz') + glob.glob(os.path.expanduser('~/Desktop/**/*wireframe*lines*.npz'), recursive=True))
    assert cand, "pass the .npz path as an argument"
    npz = cand[0]

d = np.load(npz, allow_pickle=True)
print('npz:', npz, '| keys:', list(d.keys()))
RAW = 'data/wireframe_raw'; os.makedirs(RAW + '/images', exist_ok=True)

# detect the line-coordinate grid vs the image side, then scale lines into image-pixel space
S = int(d['Xtr'].shape[1])
allc = np.concatenate([np.asarray(d['Ltr'][i]).reshape(-1, 4)
                       for i in range(len(d['Ltr'])) if len(np.asarray(d['Ltr'][i]))], 0)
lmax = float(allc.max()); GRID = 128 if abs(lmax - 128) < abs(lmax - S) else S; scale = S / GRID
print('image side=%d, line max=%.0f -> grid=%d, scaling lines x%.3f into image space' % (S, lmax, GRID, scale))

def dump(X, L, split):
    recs = []
    for i in range(len(X)):
        fn = '%s_%05d.jpg' % (split, i)
        cv2.imwrite(RAW + '/images/' + fn, X[i])                 # .jpg required by the loader
        ln = (np.asarray(L[i]).reshape(-1, 4).astype(float) * scale).tolist()
        recs.append({'filename': fn, 'lines': ln, 'height': int(X[i].shape[0]), 'width': int(X[i].shape[1])})
    json.dump(recs, open('%s/%s.json' % (RAW, split), 'w')); print(split, len(recs), 'imgs')

dump(d['Xtr'], d['Ltr'], 'train')
dump(d['Xva'], d['Lva'], 'valid')
print('done ->', RAW, '(local SSD: this takes seconds, not the 7 min Kaggle needed)')
