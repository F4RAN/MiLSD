# recon_wireframe_512.py — download the native-512 wireframe source (L-CNN / LINEA 'wireframe_processed')
# and PRINT its exact format, so the 512 npz converter can be finalized without guessing.
# Run:  python ~/Desktop/Projects/LSD-TML/local_m1/recon_wireframe_512.py [dest_dir]
import os, sys, glob, json, zipfile, urllib.request, subprocess

DEST = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/Desktop/Projects/LSD-TML/wireframe_src')
os.makedirs(DEST, exist_ok=True)
URL = 'https://github.com/SebastianJanampa/storage/releases/download/v1.0.0/wireframe_processed.zip'
ZP  = os.path.join(DEST, 'wireframe_processed.zip')

# 0) disk check
print('disk free at', DEST, ':', subprocess.run(['df', '-h', DEST], capture_output=True, text=True).stdout.splitlines()[-1])

# 1) download + unzip (only if not already there)
data = None
for c in glob.glob(DEST + '/**/wireframe_processed', recursive=True) + [os.path.join(DEST, 'wireframe_processed')]:
    if os.path.isdir(c): data = c; break
if data is None:
    if not os.path.exists(ZP):
        print('downloading wireframe_processed.zip (a few GB, one time) ...')
        urllib.request.urlretrieve(URL, ZP)
        print('downloaded', round(os.path.getsize(ZP)/1e6), 'MB')
    print('unzipping ...')
    with zipfile.ZipFile(ZP) as z: z.extractall(DEST)
    cands = glob.glob(DEST + '/**/wireframe_processed', recursive=True)
    data = cands[0] if cands else DEST
print('\n=== DATA DIR:', data)

# 2) directory tree (2 levels)
print('\n=== TREE (top 2 levels) ===')
for p in sorted(glob.glob(data + '/*')):
    isd = os.path.isdir(p); print(' ', os.path.basename(p) + ('/' if isd else ''))
    if isd:
        kids = sorted(glob.glob(p + '/*'))
        for q in kids[:6]: print('      ', os.path.basename(q) + ('/' if os.path.isdir(q) else ''))
        if len(kids) > 6: print('       ... (%d items)' % len(kids))

# 3) image count + a sample size
imgs = [f for f in glob.glob(data + '/**/*', recursive=True) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print('\n=== IMAGES: %d total ===' % len(imgs))
if imgs:
    import struct
    s = imgs[0]
    try:
        import cv2; im = cv2.imread(s); print('sample image', os.path.relpath(s, data), '-> shape', None if im is None else im.shape)
    except Exception:
        print('sample image', os.path.relpath(s, data))

# 4) annotation files + sample content (THIS is what locks the converter)
jsons = glob.glob(data + '/**/*.json', recursive=True)
print('\n=== JSON / ANNOTATION FILES ===')
for j in jsons[:12]: print('  ', os.path.relpath(j, data), '(%.1f MB)' % (os.path.getsize(j)/1e6))
npzs = glob.glob(data + '/**/*.npz', recursive=True)
if npzs: print('NPZ label files:', [os.path.relpath(n, data) for n in npzs[:6]], '...' if len(npzs) > 6 else '')

# pick the most line-relevant json and dump its shape
def peek(jp):
    try: o = json.load(open(jp))
    except Exception as e: print('  (could not parse', jp, e); return
    print('\n--- %s ---' % os.path.relpath(jp, data))
    if isinstance(o, dict):
        print('  top-level keys:', list(o.keys())[:12])
        for k in ('images', 'annotations', 'lines'):
            if k in o and isinstance(o[k], list) and o[k]:
                print('  %s[0]:' % k, json.dumps(o[k][0])[:300])
    elif isinstance(o, list) and o:
        print('  list of %d; entry[0] keys:' % len(o), list(o[0].keys()) if isinstance(o[0], dict) else type(o[0]))
        print('  entry[0]:', json.dumps(o[0])[:300])

for j in jsons:
    bn = os.path.basename(j).lower()
    if any(t in bn for t in ('train', 'valid', 'test', 'line', 'instance', 'ann')):
        peek(j)
print('\n>>> PASTE the TREE + IMAGES + JSON sample above; I will finalize the 512 converter from it.')
