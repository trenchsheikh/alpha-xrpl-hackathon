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
| 88 | hail mary | q4 max 0.995*hard_v9 (CrackiT, Sami-approved) | 49.9, Hard 0% |
| 89 | jaffa cake | w1 | 66.1 |
| 90 | last orders | q4 max 0.991*hard_v4 (CrackiT) | 55.3 |
| 91 | custard cream | w2 | 66.0 |
| 92 | bourbon | w3 | 66.0 |
| 93 | hobnob | w4 | 66.1 |
| 94 | digestive | w5 | 66.1 |
| 95 | rich tea | w6 | 65.9 |
| 96 | ginger nut | w7 | 66.0 |
| 97 | shortbread | w8 | 66.0 |
| 98 | fig roll | w9 | 66.0 |
| 99 | wafer | w10 | 66.1 |
| 100 | crumpet | v1 | 66.0 |
| 101 | scone | v2 | 66.0 |
| 102 | teacake | v3 | 66.1 |
| 103 | flapjack | v4 | 66.0 |
| 104 | cold feet | q4-2sd disagreement penalty | 65.6 |
| 105 | second opinion | mean-1sd | 65.7 |
| 106 | fence sitter | q4-1sd | 65.9 |
| 107 | maybe later | q4-0.5sd | 66.0 |
| 108 | eccles cake | v5 | 66.1 |
| 109 | bakewell | v6 | 66.0 |
| 110 | wild guess | q4+0.5sd | 65.9 |
| 111 | loose cannon | q4+1sd | 60.2 |
| 112 | maverick | q4+2sd | 22.5 |
| 113 | high five | 0.7q4+0.3max | 66.1 |
| 114 | mirror mirror | pl_300_05 (pseudo-label self-train) | 61.4 |
| 115 | echo chamber | pl_300_03 | 63.7 |
| 116 | bootstraps | pl_500_03 | 65.4 |
| 117 | self help | pl_200_03 | 63.7 |

Runs 114-117: all prec 93-94%, top1% recall Easy 50% / Med,Hard,Amb 0%, AUC .90-.91. Pseudo-label self-training hurt vs q4 (66.1). 81 attempts left.
| 118 | group hug | /workspace/crackit/rank_q4top2_best.csv (CrackiT cluster-consistency on q4) | 65.8 |
| 119 | social butterfly | h_graph-inject40 (q4 order + top 40 of LightGBM base+graph-features (lkgraph/) outside q4 top 401, inserted at q4 rank 200) | 66.2 |
| 120 | chatterbox | h_graph-rankavg (0.5 q4 rank + 0.5 graph-model rank) | 64.0 |

Run 119: prec 95%, top1% recall Easy 50% / Med 10% / Hard 0% / Amb 0%. **New best 66.2** (q4 = 66.1). Run 118 CrackiT cluster-consistency 65.8. 80 attempts left.
Run 120: prec 93%, top1% E50 M10. Full rank-avg with graph model hurts on test (64.0); small injections preferred.
| 121 | triple decker | h_graph3s-inject40 | 66.3 |
| 122 | party bus | h_graph-inject60 | 66.0 |
| 123 | late bloomer | h_graph-inject40r300 | 66.2 |
| 124 | dark horse | crackit nov_inj40 | 64.4 |
| 125 | conga line | h_graph-inject80 | 65.8 |
| 126 | front row | h_graph-inject40r100 | 66.0 |
| 127 | hail mary | crackit big_m18b40 | 62.9 |
| 128 | needle hunt | i_hardnb-inject20 | 64.5 |
| 129 | best of both | k_union | 66.3 |
| 130 | lucky dip | j_cohort-inject20 | 64.5 |
| 131 | pizza party | m_all300 | 66.1 |
| 132 | pillow fight | m_all200 | 65.7 |
| 133 | tea break (unnamed) | m_15r150 | 66.0 |
| 134 | wild card (unnamed) | r_rand20 (20 random rows at rank 380) | 65.5 |

Runs 121-134: prec / top1% E/M/H/A — 121 95% 50/0/0/0; 122 94% 50/10/0/0; 123 95% 50/10/0/0; 124 90% 40/0/0/0; 125 93% 50/10/0/0; 126 95% 50/10/0/0; 127 86% 40/0/0/0; 128 91% 40/0/0/0; 129 95% 50/0/0/0; 130 91% 40/0/0/0; 131 93% 50/10/0/0; 132 93% 50/10/0/0; 133 94% 50/10/0/0; 134 91% 40/0/0/0. **New best 66.3** (runs 121, 129). ~64 attempts left.
| 135 | night owl | s_relay40 (top 40 relay_match6h rows inserted at rank 300 of h_graph3s-inject40) | 64.2 |
| 136 | deep clean | t_deep_ra (top401 of h_graph3s-inject40, rest by h_graph-rankavg) | 65.1 |
| 137 | spring clean | t_deep_q4 (top401 of h_graph3s-inject40, rest by q4) | 66.3 |
| 138 | photo finish | f_lb1 (consensus-reordered top401 from leaderboard mapping) | 62.3 |
| 139 | final whistle | f_lb3 (top401 of h_graph3s-inject40, rest by mean rank of runs >=66.0) | 66.3 |
| 140 | last orders (may show as Run) | g_deep1 (top401 h_graph3s, rest by h_graph-inject40) | 66.3 |
| 141 | closing time | g_deep2 (top401 h_graph3s, rest by m_all300) | 66.2 |
| 142 | second wind | n_med1 (top401 from h_graph-inject40, rest by q4) | 66.2 |
| 143 | comeback kid | n_med2 (top401 from h_graph-inject40r300, rest by q4) | 66.2 |

Runs 135-143: prec / top1% E/M/H/A — 135 87% 40/0/0/0; 136 95% 50/0/0/0; 137 95% 50/0/0/0; 138 95% 50/0/0/0; 139 95% 50/0/0/0; 140 95% 50/0/0/0; 141 95% 50/0/0/0; 142 95% 50/10/0/0; 143 95% 50/10/0/0. Best remains 66.3. ~55 attempts left.
| 144 | buzzer beater | h_graph-inject30 | 66.2 |
| 145 | extra time | o_inj40 (5-seed combined LightGBM on motif+graph+cohort+medium-windows, top 40 inserted at rank 200 of h_graph3s-inject40) | 66.1 |
| 146 | penalty shootout | o_tail (top 401 of h_graph3s-inject40, rest by combined model) | 63.5 |
| 147 | long shot | p_relay20 (relay-family two-stage Hard probe, 20 rows at rank 380) | 65.3 |

Runs 144-147: prec / top1% E/M/H/A — 144 94% 50/0/0/0; 145 94% 50/10/0/0; 146 95% 50/0/0/0; 147 91% 40/0/0/0. Best still 66.3. 52 attempts left.
| 148 | musical chairs | e1_swap10 | 66.3 |
| 149 | high five | e3_graph5s (5-seed graph model, top 40 injected at rank 200 of q4) | 66.3 |
| 150 | half century | e5_inj50 | 66.2 |
| 151 | early bird | e4_inj40r150 | 66.3 |
| 152 | top shelf | e2_top100lock | 66.3 |

Runs 148-152: 148 P95% 50/0/0/0; 149 50/10/0/0 (best so far that keeps Medium; new base). Best still 66.3. 47 attempts left.
| 153 | slow burn | f2_k40r250 (e3 graph inject at rank 250) | 66.3 |
| 154 | ten pin | f3_graph10s (10-seed graph model, top 40 at rank 200) | 66.3 |
| 155 | late show | f4_med10r390 (e3 + Medium-only top 10 at rank 390) | 66.3 |
| 156 | dirty thirty | f1_k30r250 (graph K=30 at rank 250) | 66.2 |
| 157 | tail wind | f5_tailq4 (e3 top 401, rest by q4) | 66.3 |

Runs 153-157: Med top1% — 153 10; 154 10; 155 0; 156 0; 157 10. Everything stuck at 66.3. 41 attempts left.
| 158 | fresh legs | g3_graphfill350 (graph next-best fill ranks 351-401) | 66.1 |
| 159 | middle child | g1_med30r350 (e3 + Medium top 30 at rank 350) | 66.3 |
| 160 | bench press | g4_med20swap (e3 bottom 20 swapped for Medium top 20) | 66.3 |
| 161 | front row | g2_graph40r100 (graph inject at rank 100) | 66.1 |
| 162 | slow lane | g5_r250tailq4 (graph inject at 250, tail by q4) | 66.3 |

Runs 158-162: 158 P93 Med0; 159 Med10; 160 Med0; 161 Med10; 162 Med10. Lesson: touching ranks <200 or swapping many top-401 rows costs points; Medium rows at 350 neutral. 36 attempts left.
