# Team Alpha submission log

| # | Time (UK) | Run name | What changed | Holdout AP | LB AP | P@1% | R@1% |
|---|---|---|---|---|---|---|---|
| 1 | 12:47 | dev_boost catboost norelay drophard | medium-windows features, CatBoost seed 42, 2 relay feats dropped, Hard/Ambiguous positives removed from training | 51.28 | 55.6 | 0.85 | 0.27 |
| 2 | 12:55 | v01 motif catboost | + XRPL Investigator motif features (ab27e3c) | 55.85 | 59.8 | 0.90 | 0.29 |
| 3 | 12:55 | v03 motif catboost 1200it | v01 with 1200 iterations | - | 59.2 | 0.90 | 0.29 |
| 4 | 12:55 | v05 motif cat75 lgb25 | v01 rank-blended 0.75 CatBoost / 0.25 LightGBM | - | 60.3 | 0.90 | 0.29 |
| 5 | 12:58 | (unnamed) v02 cat 3 seeds | motif, rank-avg CatBoost seeds 42/7/99 | - | 60.5 | 0.91 | 0.29 |
| 6 | 12:58 | v04 motif pseudolabel | + test pseudo-labels | - | 59.9 | 0.90 | 0.29 |
| 7 | 12:58 | v06 motif relay kept | relay features kept | - | 59.7 | 0.91 | 0.29 |
| 8 | 12:58 | v07 motif depth8 | CatBoost depth 8 | - | 59.6 | 0.89 | 0.29 |
| 9 | 12:58 | v08 motif classweight5 | scale_pos_weight=5 | - | 60.6 | 0.89 | 0.29 |
| 10 | 12:58 | v09 motif cat 5 seeds | rank-avg 5 CatBoost seeds | - | 60.2 | 0.91 | 0.29 |
| 11 | 12:59 | motif-v2 cycle fix | XRPL Inv. bb8dde5, CatBoost best | - | 60.0 | 0.90 | 0.29 |
| 12 | 12:59 | motif-v2x stack detector | XRPL Inv. 2a909b6, CatBoost best | - | 60.0 | 0.92 | 0.29 |
| 13 | 13:06 | mv-scenario-agg blend | XRPL Inv. 26ee918, pattern scores spread over linked tx; 75/25 Cat/LGB | - | 63.9 | 0.93 | 0.30 |
| 14 | 13:06 | mv-windows-more blend | 8499933, extra windows | - | 61.3 | 0.93 | 0.30 |
| 15 | 13:06 | mv-ratio-tight blend | ceaf9f7, value band 0.85-1.12 | - | 60.9 | 0.91 | 0.29 |
| 16 | 13:06 | mv-ratio-wide blend | fd24719, value band 0.6-1.3 | - | 61.4 | 0.92 | 0.30 |
| 17 | 13:06 | mv-currency blend | d9bfb89, same-currency hops | - | 60.7 | 0.91 | 0.29 |
| 18 | 13:12 | prop K400 W600 a0.3 | mv-scenario-agg + linked-group mean-rank propagation | - | 59.0 | 0.90 | 0.29 |
| 19 | 13:12 | prop K200 W600 a0.2 | same, milder | - | 60.6 | 0.90 | 0.29 |
| 20 | 13:12 | prop K400 W1800 a0.3 | same, wide window | - | 48.4 | 0.67 | 0.22 |
| 21 | 13:16 | c1 scenario-agg 3seed classw | scenario-agg + CatBoost 3 seeds + spw=5 | - | 64.3 | 93% | 30% |
| 22 | 13:16 | c2 combo windows widebands classw | windows+wide bands combo + classw | - | **64.5** | 94% | 30% |
| 23 | 13:16 | d agg-wide | agg-wide branch 1824f9b, blend_b | - | 64.0 | 94% | 30% |
| 24 | 13:16 | d medium-link | medium-link 8b6bd7e, blend_b | - | 63.7 | 93% | 30% |
| 25 | 13:21 | d hub-pruned | hub-pruned aabd0c5, blend_b | - | 64.1 | 95% | 30% |
| 26 | 13:21 | e aggwide classw | agg-wide 1824f9b, blend_c | - | **65.0** | 93% | 30% |
| 27 | 13:21 | e hub classw | hub-pruned aabd0c5, blend_c | - | 64.7 | 94% | 30% |
| 28 | 13:21 | g c2 + linker propagation | c2 + CrackiT propagation (Sami approved) | 57.90 (motif base) | 65.0 | 94% | 30% |
| 29 | 13:33 | coffee run #3 | agg-wide classw + CrackiT propagation | - | **65.5** | 92% | 30% |
| 30 | 13:33 | just vibes | rank avg of c2, c1, e_aggwide, e_hub | - | 65.1 | 94% | 30% |
| 31 | 13:42 | pizza run | rank avg (c2,c1,e_aggwide,e_hub) + propagation | - | 65.5 | 94% | 30% |
| 32 | 13:42 | one more try | e_hub classw + propagation | - | 65.0 | 95% | 31% |
