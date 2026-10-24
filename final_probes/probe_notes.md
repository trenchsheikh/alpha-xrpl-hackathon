## Probe batch (executor B, 2026-10-24 ~14:30 BST), base=k2_blend.csv, 80 rows inserted at ranks 201-280 (not in base top400)
- h02_curchange_probe.csv | currency change (SendMax cur != Amount cur), larger USD first | 80 | train hard-prec@top200=0.040 (base rate 0.0056)
- h04_consolidation_probe.csv | receiver with many distinct senders, few outflows | 80 | train hard-prec@top200=0.000
- h06_roundusd_probe.csv | round USD (multiple of 10, >=50) to a receiver that forwards | 80 | flagged 216 in train, hard-prec 0.009
- h07_newwallet_fwd_probe.csv | sender first seen <1h ago and already received funds | 80 | train hard-prec@top200=0.000
- h08_cycles_probe.csv | 2-3 hop return edge (A->B->A / A->B->C->A) | 80 | train hard-prec@top200=0.030
- h09_hardmodel_probe.csv | HistGB trained on train difficulty==hard (aggregate_relay/distributed_relay/ambiguous) | 80 | OOF hard-prec@top200=0.450  <-- strongest
- h05 equal-split: no train or test rows matched (sender cv<0.15, >=3 dsts); no file
- h10: frauds outside OOF top1% are almost all frauds; their degree/amount/cycle/currency feature medians match normals -> no separating rule found


## h11-h15 hard-model blends (2026-10-24)
Base: n1_blend.csv (cheese toastie, 66.0). Hard GBM retrained on all train hard labels (probe.py feats), test scores in hardmodel_test_scores.csv.
- h11_hardblend.csv: keep base top 150, insert 60 hard at 151-210
- h12_hardblend.csv: keep base top 250, insert 100 hard at 251-350
- h13_hardblend.csv: keep base top 350, insert 150 hard at 351-500
- h14_hardblend.csv: max(rank_base,0.8*rank_hard)
- h15_hardblend.csv: 0.85 rank_base+0.15 rank_hard
- OOF (repo-agg-wide, n=40131): base AP=0.5809; h14 max(rb,.8rh) AP=0.5870; h15 .85/.15 AP=0.3230; hard-only AP(susp)=0.1177; hard-only AP(hard)=0.3986; hard prec@top200=0.470
Not submitted.
