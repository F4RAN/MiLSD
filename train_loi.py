# train_loi.py — train the LoI (Line-of-Interest) verification head on the FROZEN backbone.
#
# For each candidate line, 32 points are sampled along it, the 32-channel reduce feature (64x64) and the
# output maps are bilinearly pooled, max-pooled to 4 bins, joined with a short geometry vector, and a
# 3-layer MLP scores the line as real or spurious. Lines are re-ranked by verification x center score.
#
# Change from the first release: the LoI reads the 32-ch REDUCE map (128 KB int8) instead of the 128-ch
# conv4 map (512 KB). Keeping the conv4 map alive until after the head pushed peak SRAM to ~1.15 MB,
# over the 1 MB H7 budget. The reduce map is alive anyway, so peak memory does not change.
# Cost: -0.3 sAP10. The trained head is saved to loi_256.pt.
#
# Usage:  python train_loi.py [npz] [ckpt]
import sys, time, numpy as np, torch, torch.nn.functional as F
from milsd import load_model, prep, decode, loi_features, LoI, match_labels

NPZ = sys.argv[1] if len(sys.argv) > 1 else 'data/wireframe_lines_256_5000.npz'
CKPT = sys.argv[2] if len(sys.argv) > 2 else 'fclip_256_best.pt'
NTRAIN, TOPC, HEPOCHS, BATCH = 3000, 150, 80, 8192
torch.manual_seed(0); np.random.seed(0)

d = np.load(NPZ, allow_pickle=True); Xtr, Ltr = d['Xtr'], d['Ltr']
m = load_model(CKPT)
t0 = time.time(); FE, LA = [], []
for i in range(min(NTRAIN, len(Xtr))):                  # frozen backbone -> features are fixed, cache once
    with torch.no_grad(): out, feat = m.featfwd(prep(Xtr[i]))
    maps = out[0].numpy(); lines, sc = decode(maps)
    if len(lines) == 0: continue
    o = np.argsort(-sc)[:TOPC]; lines, sc = lines[o], sc[o]
    lab = match_labels(lines, sc, np.asarray(Ltr[i]).reshape(-1, 2, 2).astype(np.float32))
    with torch.no_grad(): FE.append(loi_features(feat, maps, lines, sc)); LA.append(torch.from_numpy(lab))
FE = torch.cat(FE); LA = torch.cat(LA)
print('cache: %d candidates (%.1f%% positive), dim %d, %.0fs' % (len(FE), 100*LA.mean(), FE.shape[1], time.time()-t0))

loi = LoI(FE.shape[1]); opt = torch.optim.Adam(loi.parameters(), 1e-3, weight_decay=1e-4)
pw = torch.tensor([(LA == 0).sum()/max((LA == 1).sum(), 1)])
for ep in range(HEPOCHS):
    loi.train(); perm = torch.randperm(len(FE))
    for k in range(0, len(perm), BATCH):
        idx = perm[k:k+BATCH]; loss = F.binary_cross_entropy_with_logits(loi(FE[idx]), LA[idx], pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
loi.eval(); torch.save(loi.state_dict(), 'loi_256.pt')
print('saved loi_256.pt (%d params). Evaluate with: python eval_milsd.py --loi' % sum(p.numel() for p in loi.parameters()))
