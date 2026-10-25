# Predicting late deliveries on Olist

INSY 674, McGill, Fall 2026. Final group project.

Fidan Shiraliyeva, Hasan Malik, Jay Malhotra, Jingyu Wu, Mia Scalabrin, Peace Isaac, Tasnim Hassan

<!-- Jay: add student IDs next to each name before submission. -->

**Research question:** Using only information available when an order is approved, how accurately can we predict which orders will arrive after the promised delivery date, and which factors contribute most to that risk?

The sections below follow the project checklist from the rubric (Géron, Appendix B). The full analysis is in `notebooks/olist_late_delivery.ipynb`.

## 1. Frame the Problem and Look at the Big Picture

### 1.1 Objective

Olist is a Brazilian marketplace that connects small sellers to large online stores. When a customer buys something, Olist gives them an estimated delivery date. Between January 2017 and July 2018, 5,986 of 85,371 delivered orders (7.0%) arrived after that date.

Late orders hurt reviews badly. They averaged 2.21 stars compared to 4.28 for orders that arrived on time, and 62.5% of them got a 1 or 2 star review (9.3% for on-time orders).

Our goal is to flag orders that are likely to be late early enough for someone to do something about it, instead of finding out from a bad review.

### 1.2 Research question

Using only information available when an order is approved, how accurately can we predict which orders will arrive after the promised delivery date, and which factors contribute most to that risk?

This has two parts. The first is how well we can rank late orders above on-time ones at approval (section 5). The second is which order, product, seller and route characteristics drive the risk (sections 4.2 and 5.4).

### 1.3 Who would use it

The main user is Olist's operations team. Every morning, the orders approved the day before get a risk score, and the riskiest 10% are flagged. For those, the team can push the seller to ship sooner, follow up with the carrier, or warn the customer and give a new date.

Seller managers could also use the scores averaged by seller to see which sellers cause the most late orders.

The model only ranks orders so people know where to look. It does not change delivery dates on its own.

### 1.4 What Olist does today

The public dataset does not document whether Olist had an internal order-level risk score. One observable safeguard is the generous promised delivery window: the median promised window is 24 days, while the median delivery actually takes 10. That works for the average order, but it treats every order the same. Even with the padding, the late rate hit 12.4% in November 2017, 14.1% in February 2018 and 19.0% in March 2018.

### 1.5 How we framed the problem

It is a supervised binary classification problem. Every past order has a known outcome (late or not), and the decision we want to support is yes or no: act on this order or leave it. We considered predicting the number of days late instead, but the team would still need to decide whether to act, so a risk ranking is more useful.

The model is trained offline and scores new orders once a day in a batch. Delivery problems play out over days, so a real-time score would not add anything. Predictions are made at `order_approved_at`, and we only use information that exists at that moment (see section 3.4).

The data is tabular, built by joining the relevant Olist order, item, product, seller, customer, geolocation, and product-category translation tables. Payments and reviews are excluded from predictive features because their availability at order approval cannot be reliably established.

### 1.6 How we measure performance

Our main metric is PR-AUC (average precision). It focuses on the late orders, which are the rare class we care about.

The metric that matches how the team would actually work is recall in the top 10%: if ops can only check 10% of orders a day, what share of all late orders do they catch? We also report it at 5% and 20%, plus ROC-AUC for comparison with other work.

Accuracy is not useful here. Predicting "on time" for every order gives 95.9% accuracy on our test set and catches no late orders at all.

Every model is scored with the same code (`src/evaluation.py`) using 5-fold time-series cross-validation on the training period, reported as mean and standard deviation. The test set is used once, at the end.

### 1.7 Baselines

| Baseline | What it is | Test PR-AUC | Test recall, top 10% |
|---|---|---:|---:|
| B0 | Random ranking (no skill) | 0.041 | 10% |
| B1 | Logistic regression on promised days and customer state | see 5.3 | see 5.3 |

B0's PR-AUC equals the test late rate. To be worth anything, the final model has to beat B1, not just B0.

### 1.8 Data

We used the Brazilian E-Commerce Public Dataset by Olist from Kaggle (CC BY-NC-SA 4.0). The raw files are in `data/raw/` and the licence is in `DATA_LICENSE.md`.

| Step | Orders |
|---|---:|
| All orders (Sept 2016 to Oct 2018) | 99,441 |
| Status is delivered | 96,478 |
| Has approval and delivery dates | 96,456 |
| Bought from January 2017 onward | 96,189 |
| Promised by 1 August 2018 | 85,371 |

The last filter is needed because the data effectively stops at the end of August 2018. An order promised for late August that showed up late would never appear as delivered, so without this filter the most recent months would look better than they really were. All of these choices are written up in `DECISIONS.md`.

| Set | Orders bought | Orders | Late rate |
|---|---|---:|---:|
| Train | Jan 2017 to Mar 2018 | 64,038 | 8.0% |
| Test | Apr 2018 to Jul 2018 | 21,333 | 4.1% |

We split by date because the model would be used on future orders. The late rate in the test period is about half the training rate, which we come back to in section 7.

## 2. Explore the Data

<!-- Hasan: attribute profile table (name, table, type, % missing, noise/outliers, distribution, usefulness, known at approval Y/N), target profile, labelled plots with a one-line takeaway each, correlations with takeaways, promising transformations, extra data that would help, summary of what we learned. -->

### 2.1 Attribute profile
### 2.2 Target
### 2.3 Key plots
### 2.4 Correlations
### 2.5 Promising transformations
### 2.6 Extra data that would help
### 2.7 What we learned

## 3. Prepare the Data

<!-- Mia: each cleaning step with rows affected and why, each feature group and why it should predict lateness, scaling and encoding choices, pipeline description, leakage audit table (column, known at approval Y/N, reason). -->

### 3.1 Cleaning
### 3.2 Feature Engineering and Table Integration
The relevant Olist tables were integrated using `scripts/build_olist_model_table.py`, producing `data/processed/olist_merged_modeling.csv`.

The merged dataset contains **96,456 delivered orders and 29 columns**, with one row per order. It combines orders, customers, order items, products, sellers, geolocation, and product-category translations.

Order items and seller information were aggregated to avoid duplicate order records. The resulting features describe purchase timing, customer and seller locations, product characteristics, prices, freight costs, and shipping distances.

Reviews were excluded from prediction because they contain post-purchase information. Payments were excluded because their availability at approval could not be confirmed. Actual delivery dates are used only to construct the target and are excluded from model inputs.

The modelling pipeline subsequently applies the project's existing date filters and chronological train/test split, reducing the dataset to **85,371 eligible orders**.

### 3.3 Scaling and encoding
### 3.4 Pipeline and leakage checks

## 4. Measure and Compare Performance

<!-- Fidan: 4.1 and 4.5. Jingyu: 4.2 and 4.3. Mia: 4.4. -->

### 4.1 Model comparison (5-fold time-series CV, mean and std)
### 4.2 Most important variables
### 4.3 Error analysis
### 4.4 Second round of feature selection and engineering
### 4.5 Shortlist

## 5. Fine-Tune the System

<!-- Tasnim: 5.1 to 5.3, test set run once. Tasnim and Jingyu: 5.4. -->

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

## 6. Conclusion

<!-- Jay: write after 5.3 is final. Answer both parts of the research question with numbers, then what ops should do with it. -->

## 7. Assumptions and Limitations

### Assumptions

1. The estimated delivery date in the data is the date the customer was shown at purchase. If not, the target measures the wrong promise.
2. Approval is the earliest point the team can act. Scoring at purchase would give them a bit more time.
3. The team can review about 10% of orders a day. Since this is a guess, we also report recall at 5% and 20%.
4. What happened between 2017 and March 2018 still holds for April to July 2018. The time split tests this directly.

### Limitations

| Limitation | Likely impact | How to address it |
|---|---|---|
| 2,963 orders never reached delivered status (shipped, cancelled, unavailable or still processing), so they have no label. | Lost or stuck parcels are probably the worst cases, so the true late rate is likely higher than 7.0%. | Count orders still marked shipped past their promise as late and rerun as a check. |
| The censoring filter drops 10,818 recent orders. | The test period ends in July 2018 and is smaller than it could be. | Use a later snapshot of the data. |
| The late rate falls from 8.0% in training to 4.1% in testing. | A cutoff chosen on training data may flag too many orders, and PR-AUC is not directly comparable between periods. | Report lift over the base rate, recalibrate on recent data, retrain and monitor monthly. |
| No carrier, holiday, weather or road distance data. | Causes of lateness that are not in the data cannot be learned. | Add a Brazilian public holiday calendar and ask for carrier data. |
| Brazil only, 2017 to 2018. | Results may not carry over to other markets or to today. | Check on current data before using it. |
| Some reviews are written before delivery. | The review gap in 1.1 is an association, not a measured cost. | Reviews are only used to motivate the problem, never as a feature. |

<!-- Jay: add model-level limitations after section 5. -->

## How to run it

Needs Python 3.12.

```bash
git clone https://github.com/JustJay7/olist-late-delivery.git
cd olist-late-delivery
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_olist_model_table.py
python -m pytest -q
jupyter lab notebooks/olist_late_delivery.ipynb
```

For individual work, download the repository and load data/processed/olist_final.csv directly into your notebook. It already contains all merges, filters, and target labels (85,371 orders, 29 columns). No additional merging or filtering is needed.

## Repo layout

```
olist-late-delivery/
├── README.md
├── DECISIONS.md          decisions we locked and why
├── CONVENTIONS.md        how we work: Git, code, leakage, plots
├── RUBRIC_CHECKLIST.md   rubric items, where each is covered, owner
├── DATA_LICENSE.md
├── requirements.txt
├── data/
│   ├── raw/                 the 9 original Olist tables
│   └── processed/
│       └── olist_final.csv
├── scripts/
│   └── build_olist_model_table.py
├── src/
│   ├── config.py         paths, dates, seeds, banned columns
│   ├── data.py           loading, target, filters, time split
│   ├── integration.py    joins approved candidate features
│   └── evaluation.py     metrics and cross-validation used by every model
├── tests/                checks for the target, split, integration and metrics
├── notebooks/            final notebook and its HTML export
├── wip/                  everyone's working notebooks
├── reports/figures/      figures for the README and slides
└── slides/
```
