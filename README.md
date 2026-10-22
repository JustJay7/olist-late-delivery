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

There is no risk score per order. Olist's safeguard is to pad every estimate: the median promised window is 24 days, while the median delivery actually takes 10. That works for the average order, but it treats every order the same. Even with the padding, the late rate hit 12.4% in November 2017, 14.1% in February 2018 and 19.0% in March 2018.

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
### 5.2 Ensembles
### 5.3 Test set results compared with the baselines
### 5.4 The final model

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
git clone <repo-url>
cd olist-late-delivery
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
jupyter lab notebooks/olist_late_delivery.ipynb
```

Run the notebook from top to bottom. Everything is rebuilt from the files in `data/raw/`.

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
│       └── olist_merged_modeling.csv
├── scripts/
│   └── build_olist_model_table.py
├── src/
│   ├── config.py         paths, dates, seeds, banned columns
│   ├── data.py           loading, target, filters, time split
│   └── evaluation.py     metrics and cross-validation used by every model
├── tests/                checks for the target, split and metrics
├── notebooks/            final notebook and its HTML export
├── wip/                  everyone's working notebooks
├── reports/figures/      figures for the README and slides
└── slides/
```
