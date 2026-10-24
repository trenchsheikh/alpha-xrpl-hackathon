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
| 33 | 13:55 | trust the process | rank avg of g_aggwide-prop, g_rankavg-prop, g_c2-prop | - | **65.6** | 93% | 30% |
| 34 | 14:15 | (name not saved) | k1_blend: 0.5 g_aggwide-prop + 0.5 g_f_rankavg-top-prop | - | 65.7 | - | - |
| 35 | 14:15 | tea break | k2_blend: 0.4 aggwide-prop + 0.4 rankavg-top-prop + 0.2 g_e_hub-classw-prop | - | **65.8** BEST | - | - |
| 36 | 14:15 | lucky socks | k3_blend: 0.7 h_avg3prop + 0.3 aggwide-prop | - | 65.6 | - | - |
| 37 | 14:15 | hold my coffee | k4_blend: geo-mean ranks of aggwide-prop, rankavg-top-prop, c2-prop | - | 65.6 | - | - |

Note: g_scenlink-prop rejected as duplicate of coffee run #3 (not submitted).
| 38 | m1_blend (0.35 aggwide-prop+0.35 rankavg-top-prop+0.3 hub-prop) | snack attack | 65.8 |
| 39 | m2_blend (+0.2 c2-prop) | rubber duck | 65.7 |
| 40 | m3_blend (0.4/0.3/0.3) | second breakfast | 65.8 |
| 41 | m4_blend (0.5 k1+0.5 k2) | it's not a bug, it's a feature | 65.8 |
| 42 | g_scenlink2-prop (scen-link rebuilt with 29 lk_* cols + prop) | plot twist | 65.8 |
| 43 | e_scenlink2-classw (no prop) | banana bread | 65.5 |
| 44 | n1_blend (0.5 scenlink2-prop + 0.5 k2) | cheese toastie | 66.0 BEST |
| 45 | n2_blend (0.7 scenlink2-prop + 0.3 k2) | nap time | 66.0 |
| 46 | q1 (0.6 scenlink2-prop+0.4 k2) | pancake stack | 66.0 |
| 47 | q2 (0.4 S+0.3 k2+0.3 h_avg3prop) | sunday roast | 65.9 |
| 48 | q3 (0.5 S+0.2 aggwide+0.15 rankavg+0.15 hub) | biscuit tin | 66.0 |
| 49 | q4 (0.4 S+0.1 e_scenlink2+0.5 k2) | fish and chips | 66.1 BEST |
| 50 | p1_oofblend (XRPL max-of-ranks agg/hub/medium) | lemon drizzle | 64.3 |
| 51 | r1 (p1 + prop) | spicy noodles | 64.5 |
| 52 | mango lassi | r2_blend max-rank S,K2 | 65.9 |
| 53 | blue moon | s1 | 64.4 |
| 54 | tiny hat | s2 | 63.0 |
| 55 | secret sauce | h09 hard probe | 59.8 |
| 56 | big cheese | h14 max(rank, hardmodel) on n1 | 66.1 (Med 10%) |
| 57 | jelly bean | h11 | 60.7 |
| 58 | hot potato | h12 | 60.4 |
| 59 | wiggle room | h13 | 60.0 |
| 60 | wobbly table | s3 | 60.5 |
| 61 | lost sock | s4 | 64.6 |
| 62 | paper plane | s5 | 63.3 |
| 63 | rainy day | s6 | 64.4 |
| 64 | lucky dip | s7 | 63.1 |
| 65 | night owl | s8 | 63.1 |
| 66 | coin flip | h02 | 60.1 |
| 67 | merry-go-round | h08 | 60.7 |
Note: Hard recall 0% on every insertion; insertions cost precision. ~131 attempts left.
| 68 | duck soup | x1_experts | 66.0 |
| 69 | jam sandwich | x2_experts | 65.8 |
| 70 | party hat | x3_experts | 50.2 |
| 71 | toast soldiers | y1 rank avg h14+x1 | 66.1 |
| 72 | long shot | z1 hard model only (diagnostic) | 3.9, Hard 0% |
| 73 | cup of tea | x4_medonly | 66.0 |
| 74 | wild card | z2 medium model only (diagnostic) | 7.3, Med 10% |
| 75 | golden hour | t1 geo-mean top5 | 66.0 |
| 76 | slow cooker | t2 min-rank top5 | stuck processing |
| 77 | middle child | t3 median top5 | 66.0 |
| 78 | grumpy cat | r2_consmin | 66.0 |
| 79 | group hug | r1_consensus | 66.0 |
| 80 | sleepy panda | r5_harm | 66.1 |
| 81 | big breakfast | r4_q4heavy | 66.1 |
| 82 | window seat | u1 q4+mv-windows | 65.9 |
| 83 | middle lane | u2 q4+medium-link | 65.9 |
| 84 | three amigos | u3 q4+avg3prop | 66.0 |
| 85 | fizzy pop | r6_q4y1 | 66.1 |
| 86 | top dog | r7_q4h14max | 66.1 |
| 87 | lucky seven | r8_q4y1_73 | 66.1 |
