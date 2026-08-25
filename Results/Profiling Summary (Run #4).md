# TML4 Profiling — Run 4 (expanded grid + richer metrics, all 8 datasets)

Latest full profiling run. Two things changed since last time: the **grid grew from 176 to 275 combinations** (a new preprocessor and a new feature map), and evaluation now reports **balanced accuracy, AUC, sensitivity, and specificity** rather than raw accuracy alone. 5-fold CV, 300s timeout, on the SCC.

## Bottom line

The two new pipeline options are already earning their place (they win 3 of the 8 datasets), the richer metrics give a more honest read on performance, and **Hist Gradient Boosting no longer times out anywhere** — the only remaining timeout is classic Gradient Boosting on GCM's full-gene-count combos. Total compute actually dropped despite 100 more combinations.

## What's new this run

- **Grid: 176 → 275** (5 preprocessors × 5 feature maps × 11 classifiers).
- **New preprocessor — "Log2 + standardize."** Wins DLBCL outright (balanced accuracy 0.991, AUC 0.993).
- **New feature map — "Select best 500 genes"** (univariate feature selection). Wins Prostate1 and Prostate2, and — by cutting 16k genes down to 500 — lets Gradient Boosting and Hist GB finish on combos that used to time out.
- **Richer metrics.** Every combo now reports balanced accuracy, AUC, sensitivity, and specificity. Balanced accuracy (the average of sensitivity and specificity) is the honest headline number for imbalanced response data — it can't be inflated by predicting the majority class the way raw accuracy can.

## Best genuine result per dataset (by balanced accuracy; excludes unreliable combos)

| dataset   | bal. acc | AUC   | winning combination                                   |
|-----------|---------:|------:|-------------------------------------------------------|
| Lung      |    1.000 | 1.000 | identity → identity → Logistic Regression             |
| Prostate3 |    1.000 | 1.000 | identity → Class-conditional PCA → Nearest Shrunken Centroid |
| DLBCL     |    0.991 | 0.993 | **Log2 + standardize** → identity → SVM (linear)      |
| Leukemia  |    0.980 | 1.000 | identity → identity → Naive Bayes                     |
| Prostate1 |    0.942 | 0.965 | identity → **Select best 500** → Hist Gradient Boosting |
| Colon     |    0.907 | 0.857 | identity → PCA (50) → SVM (RBF)                        |
| GCM       |    0.837 | 0.898 | identity → Rank transform → Logistic Regression       |
| Prostate2 |    0.808 | 0.847 | identity → **Select best 500** → Random Forest        |

SVM (RBF) remains healthy after the earlier `gamma` fix (it wins Colon). The strong performers are now spread across Logistic Regression, linear SVM, SVM (RBF), Naive Bayes, K-TSP, Random Forest, Hist GB, and Nearest Shrunken Centroid — no single classifier dominates, which is expected for high-dimensional genomics.

Same caveat as always on the perfect 1.000 scores (Lung, Prostate3, and Leukemia's AUC): on small, very wide datasets these signal an easy benchmark, not a flawless model. AUC alongside balanced accuracy makes this easier to judge — where both are 1.000 on ~30–180 patients, treat it as "the pipeline works," not "the model is perfect."

## Timeouts — Hist GB fixed, only classic GB remains

- **Timeouts dropped 16 → 9, and all 9 are classic Gradient Boosting on GCM** (the full-gene-count feature maps: identity and rank-transform).
- **Hist Gradient Boosting now times out nowhere** — the new "Select best 500 genes" map, plus the existing PCA maps, keep its dimensionality low enough to finish. This effectively resolves the Hist GB scaling concern from earlier runs.
- Remaining action is narrow: classic GB still can't handle ~16k genes directly. Either drop classic-GB-on-full-gene-count combos, or rely on the feature-selection maps (which already fix it in practice).

## Reliability

40 combos across all datasets were flagged **unreliable** (balanced accuracy at or near chance despite converging). These are heavily concentrated: **35 of 40 are Class-conditional PCA** feeding weak learners (Gradient Boosting, Decision Tree), landing at ~0.48–0.51 balanced accuracy. They're flagged and excluded from Auto's pick, so they can't mislead — but Class-conditional PCA is the clear candidate if the team wants to prune a persistently weak feature map.

## Compute and scaling

- **Total single-core work: ~204 min** — *down* from 289 despite 100 more combinations. The feature-selection map makes most classifier fits far cheaper (500 genes vs 16k), and Hist GB no longer runs away.
- **Parallel efficiency: 87–91% at 8 cores, 73–88% at 16** — the best scaling we've seen, and consistent across datasets. 8 cores remains the sweet spot; no GPUs needed.

## Where things stand

- **Fixed / resolved:** SVM (RBF) collapse, AdaptivePCA small-cohort failures, Hist GB timeouts, and evaluation now uses honest imbalance-aware metrics.
- **Working well:** parallel scheduling (near-ideal to 8 cores), the two new pipeline options, total runtime in minutes.
- **Still open (narrow):** classic Gradient Boosting timing out on GCM's full-gene combos; optionally pruning Class-conditional PCA, which produces most of the remaining unreliable results.
