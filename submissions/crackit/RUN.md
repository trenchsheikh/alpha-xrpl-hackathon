# c2 + linker propagation
1. Build the c2 test predictions with blend_c.py (see the repo root) and write out/sub/c_c2.csv.
2. Train the pair linker: `python crackit/linker.py` (out-of-fold pair model; writes /tmp/lk_*).
3. Apply propagation: `python crackit/prop_apply.py test out/sub/c_c2.csv submission.csv`
   Score: x = 0.4*rank(own) + 0.6*cluster max (cap 15, min link 0.5); final = 0.85*x + 0.15*neighbour mean over links > 0.7.
