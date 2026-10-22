# Decision log

Locked decisions for the project. Each one is implemented in `src/config.py` and `src/data.py`.
To change a decision, raise it with Jay first, then update this file, the config and
the README in the same commit. All figures below were produced by `src/data.py` on the raw data.

| ID | Decision | Choice | Why |
|---|---|---|---|
| D1 | Unit of analysis | One row per order | The intervention (escalate, nudge seller, warn customer) happens per order. Item and seller tables are aggregated to order level. Payment data is excluded from current predictive features pending confirmation of approval-time availability. |
| D2 | Population | Status `delivered`, has approval and delivery dates, purchased from 2017-01-01 | Only delivered orders have a label. 2016 holds a few hundred orders and is too sparse. |
| D3 | Target | `is_late = 1` if delivered on a later **calendar day** than the promised date | The promised date has no time of day (always 00:00). Comparing timestamps would mark an order delivered on the promised day as late, which inflates the late rate from 6.8% to 8.1%. |
| D4 | Censoring rule | Drop orders promised after 2018-08-01 (snapshot 2018-08-31 minus 30 days) | Recorded deliveries collapse after 30 Aug 2018, so late orders promised near the end are never seen as delivered. 95% of late orders arrive within 31 days of the promise. Without this rule, the most recent months look artificially on time. |
| D5 | Prediction moment | `order_approved_at` | Ops can act from approval onward. Anything recorded after approval (carrier pickup, delivery, status, reviews) is banned as a feature. See `config.BANNED_FEATURE_COLUMNS`. |
| D6 | Train/test split | Train: purchases before 2018-04-01. Test: 2018-04-01 onward | The model would be used on future orders, so it must be tested on future orders. A random split lets the model learn from the future. |
| D7 | Cross-validation | 5-fold `TimeSeriesSplit` on the training period via `src.evaluation.cv_splitter()`; report mean and std | Keeps every validation fold in the future of its training fold, matching D6. |
| D8 | Metrics | Primary PR-AUC; operating metric recall in the top 10% riskiest orders; secondary ROC-AUC | The class is rare (7%) and ops can only act on a limited share of orders. Accuracy is misleading: predicting "on time" for every order scores 95.9% on the test set. |
| D9 | Baselines | B0 no-skill (always predicts the base rate); B1 logistic regression on promised days and customer state | B0 sets the floor. B1 tests whether the full model beats the two most obvious signals. |
| D10 | Feature history | Seller and route history may only use orders **delivered before** the current order's purchase time | Otherwise an order's own outcome, or later outcomes, leak into its features. |
| D11 | Methods | Only methods covered in INSY 674 materials | Course requirement. |
| D12 | Data source | Olist Kaggle release (2021-10-01), committed as gzip CSVs in `data/raw/` | Reproducible from the repo alone. See `DATA_LICENSE.md`. |

## Population funnel

| Step | Orders |
|---|---:|
| Raw orders | 99,441 |
| Status `delivered` | 96,478 |
| Has approval and delivery dates | 96,456 |
| Delivery not before purchase | 96,456 |
| Purchased from 2017-01-01 | 96,189 |
| Promised on or before 2018-08-01 (censoring rule) | 85,371 |

Late orders in the final population: 5,986 (7.0%).

## Split

| Set | Purchase dates | Orders | Late | Late rate |
|---|---|---:|---:|---:|
| Train | 2017-01-05 to 2018-03-31 | 64,038 | 5,110 | 8.0% |
| Test | 2018-04-01 to 2018-07-27 | 21,333 | 876 | 4.1% |

The late rate halves between train and test. The training period includes the November 2017
(12.4%) and February to March 2018 (14.1%, 19.0%) spikes; the test period does not. This drift is
real and is reported, not hidden: PR-AUC depends on the base rate, so results are also reported
as lift over the base rate.
