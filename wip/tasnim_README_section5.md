## 5. Fine-Tune the System

All numbers come from `wip/tasnim_tuning.ipynb`. Cross-validation uses the shared 5-fold time-series split with a 2,000-order gap (`src/evaluation.py`), on the training period only. The test set is read in one place, after the final model was locked.

### 5.1 Hyperparameter search

We tuned the three shortlisted models from 4.5 with randomized search (`RandomizedSearchCV`, `refit="pr_auc"`): 40 trials for logistic regression, 20 for Random Forest and 30 for HistGradientBoosting.

We chose random search over grid search because each model has 5 to 7 hyperparameters, several of them continuous. For the same budget, random search tries many more distinct values of the parameters that matter (Bergstra and Bengio, 2012). Scale parameters such as C, learning rate and L2 were sampled on a log scale.

Following the checklist, data-preparation choices were tuned as hyperparameters too:
- the rare-category threshold of the one-hot encoder
- class weighting
- native categorical splits for HistGB
- dropping `purchase_month`

Before tuning, the tree models overfit heavily. Their lift on training folds was 6 to 12 times their lift on validation folds, against 1.5 for logistic regression. The tree search spaces therefore focused on regularisation.

| Model | Selected settings | CV PR-AUC | CV lift | Recall top 10% |
|---|---|---:|---:|---:|
| Logistic regression, untuned | C = 1 | 0.158 ± 0.122 | 1.81 ± 0.21 | 21.5% |
| Logistic regression, tuned | L1, C = 0.157, rare categories < 20 grouped | 0.161 ± 0.124 | 1.87 ± 0.21 | 22.5% |
| **Logistic regression, tuned, without `purchase_month`** | as above | **0.168 ± 0.121** | **1.98 ± 0.24** | **23.7%** |
| Random Forest, tuned, without `purchase_month` | depth 12, ≥ 18 orders per leaf, balanced_subsample | 0.161 ± 0.127 | 1.80 ± 0.30 | 20.7% |
| HistGB, tuned, without `purchase_month` | learning rate 0.014, 395 trees, ≥ 227 orders per leaf, L2 4.9 | 0.157 ± 0.118 | 1.79 ± 0.28 | 21.5% |

The PR-AUC standard deviation is large because the late rate of the validation folds ranges from 2.8% to 18.8%. Lift (PR-AUC divided by the fold's late rate) removes that effect.

Regularisation more than halved HistGB's overfitting, and every model gained a little from tuning. Across all 90 trials, though, validation lift stayed in a narrow band whatever the training fit. Extra model capacity bought memorisation, not better predictions. Native categorical splits for HistGB did not help, so one-hot encoding was kept.

**Dropping `purchase_month` did more than all the tuning.** The training data covers only 15 months, so January to March appear in both years and every other month appears once. The same month carried very different risk: February was 3.0% late in 2017 and 14.1% in 2018, March 4.6% and 19.0%. The column records which period was bad, not seasonality.

We fixed a rule before testing: keep the month only if it helps on at least 4 of 5 folds by more than one standard error. It did not, for any model. Dropping it raised CV PR-AUC by 0.007 to 0.012 and reduced overfitting for all three models. The gain was concentrated in the fold covering January to mid-February 2018. The model had only seen those months in 2017, when they were about 3% late, so `purchase_month` told it they were safe just as they became risky. Real seasonal effects would be better captured by a holiday or peak-season flag.

### 5.2 Ensembles

Jingyu's error analysis (4.3) showed the three models catch partly different late orders, so we tested four voting ensembles of the tuned models (all without `purchase_month`):
- soft voting of all three
- soft voting of LR and HistGB
- weighted soft voting
- rank averaging

We did not use stacking. Its meta-model needs out-of-fold predictions on folds that partition the data, which time-series folds do not. A shuffled split would let it learn from future orders.

The selection rule was fixed in advance: a more complex model replaces logistic regression only if it is better on at least 4 of 5 folds and its average gain is larger than one standard error. The best ensemble, the weighted vote, only tied logistic regression (CV PR-AUC 0.168 for both). It was better on 3 of 5 folds and had lower lift and lower recall in the top 10%. No ensemble passed.

**The final model is the tuned logistic regression without `purchase_month`.** It is also the simplest model, fits in about a second, and can be explained to the ops team.

### 5.3 Test set results compared with the baselines

Test period April to July 2018: 21,333 orders, 876 late (4.1%). As in cross-validation, a 2,000-order gap separates training from testing: the final model and the baselines are trained on orders up to 22 March 2018, because the outcomes of the last 2,000 orders before 1 April would not yet have been known.

| Model | PR-AUC | Lift | ROC-AUC | Recall top 5% | Recall top 10% | Recall top 20% | Precision top 10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Final: tuned logistic regression** | **0.126** | **3.07** | **0.761** | **22.0%** | **36.1%** | **54.0%** | **14.8%** |
| B1: promised days + customer state | 0.109 | 2.66 | 0.733 | 20.1% | 33.6% | 50.2% | 13.8% |
| B0: random | 0.042 | 1.01 | 0.492 | 4.9% | 10.2% | 19.6% | 4.2% |

*The gap was applied in review, after a first test run without it. That run gave PR-AUC 0.131 and caught 328 late orders (B1: 299); with the gap the figures are 0.126 and 316 (B1: 294). The model and its settings did not change, and the gap made the score slightly worse. Both runs are in the notebook (5.3.1).*

The final model beats both baselines on every metric. In a paired bootstrap of the test set:
- PR-AUC gain over B1: +0.017 (95% CI 0.006 to 0.027)
- recall gain in the top 10%: +2.5 points (95% CI -0.4 to 4.8)

The PR-AUC gain is clearly real: the full model ranks late orders better overall. The recall gain at the 10% budget is positive but within the noise, since its interval includes zero. Promised days and customer state already carry most of the signal.

Test lift (3.1) is higher than CV lift (2.0), so there is no sign of overfitting. The training period contains the spikes of November 2017 and February to March 2018, when almost every order was at risk and lateness was hard to rank.

The model beats B1 on both metrics in April and May. May 2018 had the highest late rate (6.6%) and a weak lift (2.3), which coincides with Brazil's nationwide truck drivers' strike, a shock no feature in the data can see. June and July are mixed, with B1 ahead on one metric in each, but each month has fewer than 75 late orders, so those comparisons are noisy.

After locking the final model, we also scored the other candidates on the test set for transparency. The LR + HistGB vote had a slightly higher PR-AUC (0.132) and the same recall in the top 10% (36.1%). We did not switch models based on the test set.

### 5.4 The final model

The final model is logistic regression with L1 regularisation (C = 0.157), no class weights, rare categories (< 20 orders) grouped and `purchase_month` removed, trained on 62,038 orders. L1 switched off 65 of the 124 encoded inputs, so 59 inputs carry the signal.

Measured by test-set permutation importance, the strongest drivers are:
1. **Promised days.** One standard deviation longer roughly halves the odds of being late (odds ratio 0.52).
2. **Distance.** Odds ratio 1.55.
3. **Customer state.**
4. **Freight cost.** Odds ratio 1.35.
5. **Same-state shipping share.** Odds ratio 0.79.

This matches Jingyu's validation-fold analysis in 4.2. These are associations used for ranking, not causes.

If ops review the riskiest 10% of orders each day (about 18 orders), they would have caught **316 of the 876 late orders** in the test period, against 294 with B1 and about 88 with random checks. 14.8% of flagged orders were late, 3.6 times the rate of a random order. The advantage over random checks is large and certain; the advantage over B1 at this budget is small.

The scores should be used as a ranking, not as probabilities. The model learned from a period with twice the late rate, so its mean predicted risk on the test set is 8.1% against an observed 4.1%. A cut-off chosen on training data would flag 10.9% of test orders instead of 10%. Deployment should flag the top 10% each day. If real probabilities are ever needed, the scores should first be recalibrated on recent data.

<!-- Jay: model-level rows for section 7 -->

| Limitation | Likely impact | How to address it |
|---|---|---|
| Every model family plateaus at about 1.8 to 2.0 CV lift | 64% of late orders are still missed when reviewing 10% of orders | Add approval-time features: seller's past on-time rate (allowed by D10), carrier, holiday or peak-season flags, weather and strike alerts |
| At the 10% budget, the gain over B1 is within the noise | The extra features may not change what ops catch at this budget | Confirm on a fresh period; if it holds, B1 is a credible lightweight fallback |
| Scores are inflated about 2x on the test period | A fixed threshold would flag the wrong number of orders as the late rate changes | Flag the top 10% daily; recalibrate monthly if probabilities are needed |
| Gains from tuning and ensembles are within the CV noise | The choice among the top few models is uncertain | Track monthly lift and recall in the top 10%; re-compare LR and the LR + HistGB vote on new data |
| PR-AUC averages are dominated by the high-late-rate fold | Selection could favour models that do well in spike months | We reported lift alongside, and it gives the same choice |
| One four-month test window that includes an external shock | Test results carry real uncertainty (bootstrap CIs above) | Retrain quarterly and monitor monthly |
