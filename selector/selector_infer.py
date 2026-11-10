"""Selector inference: pick 1 of K sampled trajectories per scene with the trained GBM.

Run from the AutoVLA root, after eval_rfs_bestofk.py wrote its npz and
train_selector2.py saved the model:

    python selector_infer.py --npz eval_bestofk_results.npz --model selector_gbm_merged.pkl

Writes the picked trajectory per scene to --out. Prints RFS for the GBM pick, the
medoid pick and the oracle when the npz holds per-candidate RFS (per_cand).
"""
import os, sys, pickle, argparse, joblib, numpy as np
DATA = os.environ.get('WOD_E2E_ROOT', './dataset/waymo_e2e')
# same feature function as in training
exec('def features' + open('train_selector.py').read().split('def features')[1].split('# ---- train')[0])

p = argparse.ArgumentParser()
p.add_argument('--npz', default='eval_bestofk_results.npz')
p.add_argument('--model', default='selector_gbm_merged.pkl')
p.add_argument('--out', default='selector_picks.npz')
args = p.parse_args()

zv = np.load(args.npz, allow_pickle=True)
names, trajs, lps = list(zv['names']), zv['preds4hz'], zv['logprobs']   # [B,K,20,2], [B,K]
rated = {e['name']: e for e in pickle.load(open(os.path.join(DATA, 'rated_val.pkl'), 'rb'))}
past = np.stack([rated[n]['past'] for n in names])
sp = np.linalg.norm((past[:, -1] - past[:, -2]) / 0.25, axis=1)        # initial speed [B]
B, K = trajs.shape[:2]

gbm = joblib.load(args.model)
pred = gbm.predict(features(trajs, lps, sp).reshape(B * K, -1)).reshape(B, K)
pick = pred.argmax(1)
np.savez(args.out, names=np.array(names), pick=pick, traj=trajs[np.arange(B), pick])
print(f'{B} scenes, K={K} -> saved picks to {args.out}')

if 'per_cand' in zv.files:
    per = zv['per_cand']                                                # [B,K] real RFS
    pw = np.linalg.norm(trajs[:, :, None] - trajs[:, None, :], axis=-1).mean(-1)
    print(f'GBM selector : {per[np.arange(B), pick].mean():.3f}')
    print(f'medoid       : {per[np.arange(B), pw.sum(-1).argmin(1)].mean():.3f}')
    print(f'oracle       : {per.max(1).mean():.3f}')
    print(f'mean sample  : {per.mean():.3f}')
