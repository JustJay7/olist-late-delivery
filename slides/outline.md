# Slide outline: Predicting late deliveries on Olist

Owner: Peace. Draft v3, 26 Oct 2026 (v3 rebuilt against the final integrated README). Everything here is still open to change; comments welcome in the group chat.

**Time limit:** 10 minutes. **Target:** about 9:20 of content, which leaves room for speaker handovers. 13 slides (including the title) plus backup.

**Storyline in one sentence:** late orders destroy customer satisfaction, Olist's padded promises don't prevent them, and a simple model scored at approval catches 36% of late orders by checking just 10% of orders (3.6× random checks). The next gains need new data, not more modelling.

**Rules for the deck**
- Every slide headline is the takeaway, written as a full sentence, not a topic label.
- At most one main visual per slide, with a title, labelled axes and units.
- Every number comes from the final README / notebooks (CONVENTIONS §6). Section references in brackets, e.g. (README 5.3).

**Status key:** ✅ content final · 🎨 visual still to build or restyle

---

## Slide-by-slide

### 0. Title (0:10) ✅
- Predicting Late Deliveries at Approval Time: Olist Brazilian E-Commerce
- Team names, INSY 674, Fall 2026

### 1. Late orders turn happy customers into 1-star reviews (0:50) ✅ 🎨
*Rubric: Objective (the problem and why it matters)*
- 1 in 14 delivered orders arrived late (5,986 of 85,371, 7.0%).
- Late orders average **2.21★ vs 4.28★** on time, and **62.5%** get a 1–2★ review (vs 9.3% on time).
- **Visual (F1):** side-by-side bars of the share of 1–2★ reviews, on time vs late.
- *Say aloud:* reviews only motivate the problem; they are never model inputs.
- Source: README 1.1.

### 2. Ops needs a daily shortlist of risky orders (0:40) ✅ 🎨
*Rubric: Objective (who it serves and how it's used)*
- User: Olist's operations team. Each morning, yesterday's approved orders get a risk score and the riskiest 10% (about 18 orders a day) are flagged.
- Actions: push the seller to ship, chase the carrier, or warn the customer with a new date.
- The model ranks orders for people to review; it doesn't change dates by itself.
- **Visual (F2):** flow diagram: order approved → nightly risk score → top 10% flagged → ops action.
- Source: README 1.3, 5.4.

### 3. Padding the promise isn't enough: lateness comes in waves (0:45) ✅
*Rubric: Objective (current workaround) and an interesting finding*
- The median promise is **24 days**, but the median delivery takes **10**. Every order gets the same padding.
- Late rate still spiked: **12.4%** (Nov 2017, Black Friday volume), **14.1%** (Feb 2018), **19.0%** (Mar 2018). Those three months hold **33% of training orders but 62% of the late ones**.
- Feb–Mar 2018 was network-wide: even São Paulo's late rate nearly tripled (3.4% → 9.4%).
- **Visual:** Hasan's `reports/figures/02_monthly_volume_and_late_rate.png` (ready).
- Source: README 1.4, 2.2.

### 4. Olist's public data lets us predict at the moment ops can act (0:50) ✅ 🎨
*Rubric: Objective (how the project and dataset meet it)*
- Dataset: Olist Kaggle release, 9 joined tables (orders, items, products, sellers, customers, geolocation).
- Framing: supervised binary classification (late or not), scored offline in a daily batch, **predicting at order approval** using only information available then.
- Funnel: 99,441 orders → **85,371** usable (delivered, dated, 2017 onward, censoring rule).
- Metrics: **PR-AUC**, plus **recall in the top 10%** ("of all late orders, how many does ops catch?"). Accuracy is useless: predicting "on time" for everything scores 95.9% on the test set and catches zero late orders.
- **Visual (F4):** population funnel bar (left) plus one-line metric definition (right).
- Source: README 1.5–1.8.

### 5. Our process: the checklist end to end, with no peeking at the future (0:40) ✅ 🎨
*Rubric: Process (end-to-end, concise)*
- **Visual (F5):** horizontal pipeline: Frame → Explore → Prepare → Compare 7 models → Tune + ensembles → Test once.
- Annotate underneath:
  - Time split: train Jan 2017–Mar 2018 (64,038 orders) | test Apr–Jul 2018 (21,333 orders).
  - One leak-free scikit-learn pipeline (fill, log, scale, one-hot) refitted inside every fold; 20 features.
  - A 2,000-order gap (about two weeks) before every validation fold and the test set, because those orders' outcomes wouldn't have been known yet.
- *Say aloud:* "A random split would have flattered us: 27.7% recall instead of 23.7% in cross-validation." (README 4.4)
- Source: README 3, 4.4, DECISIONS D6–D7.

### 6. Late deliveries happen when the promise is too short for the distance (0:55) ✅
*Rubric: Process and Insights (interesting findings)*
- Orders travelling **over 889 km with a promise of 22 days or less: about 1 in 5 late** (20.1–20.7%).
- Orders **under 137 km with a promise of 30+ days: 1.7% late**.
- Olist already gives far customers longer promises, just not long enough on the longest routes.
- Callout: **where it goes matters more than what's in it.** Rio de Janeiro is late 15.6% of the time, 2.9× Minas Gerais (5.3%), despite shorter distances and the same promise. Product categories only range from 5.9% to 10.2%.
- **Visual:** Hasan's `reports/figures/05_distance_x_promise_heatmap.png` (ready). The state chart goes in backup.
- Source: README 2.3, 2.7.

### 7. Simple beats complex: logistic regression led from the start (0:45) ✅ 🎨
*Rubric: Process and Insights (what worked and what didn't)*
- 7 models from 5 families, 5-fold time-series CV, compared on **lift** (PR-AUC ÷ the fold's late rate), because fold late rates swing from 2.8% to 18.8%.
- Lift: logistic regression **1.81**, B1 (2 features) 1.76, random forest 1.61, gradient boosting 1.54, naive Bayes 1.37, decision tree 1.25, random 1.00.
- Trees overfit: their training lift was 6–12× their validation lift.
- **We expected trees to win** (the EDA showed curves and interactions). **They didn't.**
- **Visual (F7):** horizontal bars of lift ± std per model, with B0 and B1 as reference lines.
- Source: README 4.1, 4.5, 2.7.

### 8. Dropping one misleading column helped more than all the tuning (0:45) ✅ 🎨
*Rubric: Process (fine-tuning) and Insights (what worked and what didn't)*
- Randomized search on the 3 shortlisted models (LR, random forest, HistGB; 90 trials). CV lift for logistic regression: **1.81 untuned → 1.87 tuned → 1.98 without `purchase_month`**.
- Why: 15 months of data, so the month column remembered *which* months had been bad, not seasonality. February was **3.0%** late in 2017 and **14.1%** in 2018.
- **What didn't work:** 4 voting ensembles (best one only tied LR) and 8 extra features (recent late rates, seller history, route slack, promise per 100 km).
- Final model: tuned logistic regression (L1, `purchase_month` removed). It's the best, the fastest, and the easiest to explain.
- **Visual (F8):** simple step chart of CV lift: untuned 1.81 → tuned 1.87 → no month 1.98, with ensembles and extra features shown as "no gain".
- Source: README 4.4, 5.1, 5.2.

### 9. Checking the riskiest 10% of orders catches 36% of late deliveries (0:55) ✅ 🎨
*Rubric: Fine-tuning result and the objective payoff*
- Test set, Apr–Jul 2018 (21,333 orders, 876 late), touched once:
  - **Final model: 316 of 876 late orders caught (36.1%)**
  - B1 (2-feature baseline): 294 (33.6%)
  - Random checks: about 88 (10.2%)
- Flagged orders were late **3.6× as often** as a random order. PR-AUC 0.126 vs 0.109 (B1) vs 0.042 (random).
- Honest caveat: the PR-AUC gain over B1 is real (95% CI +0.006 to +0.027), but at the 10% budget the gain is within the noise (+2.5 points, CI −0.4 to +4.8). Promised days and customer state already carry most of the signal.
- **Visual:** right panel of Tasnim's `reports/figures/tasnim/5_3_test_pr_and_budget_curves.png` (recall at review budget). Restyle to match Hasan's colours and highlight the 10% line.
- Source: README 5.3, 5.4, 6.

### 10. Promise and distance drive the risk; "ordinary" late orders stay invisible (0:50) ✅ 🎨
*Rubric: Insights (memorable takeaway) and error analysis*
- Test-set drivers (permutation importance): **promised days** (one std longer roughly halves the odds, OR 0.52), **distance** (OR 1.55), **customer state**, freight (OR 1.35), same-state share (OR 0.79). Price, weight, category and purchase time barely matter.
- What the model misses: the late orders it doesn't catch look ordinary (shorter, 727 vs 1,011 km, more often within one state). All three shortlisted models miss the **same 66%** of late orders.
- What a person would have needed: whether the seller handed over to the carrier on time, carrier backlogs, strikes or holidays, and the current network-wide delay.
- **Visual:** Tasnim's `reports/figures/tasnim/5_4_permutation_importance_test.png`, restyled, top 5–7 only.
- Source: README 4.3, 5.4.

### 11. Assumptions and limitations, and how we'd fix them (0:50) ✅ 🎨
*Rubric: Assumptions and Limitations (specific, with impact and fix)*
- **Assumptions (one line each):** the promised date is the one shown at checkout; ops can review about 10% of orders a day (so we also report 5% and 20%); 2017–Mar 2018 patterns still hold in Apr–Jul 2018.
- **Visual (F11):** 3-column table (Limitation | Impact | Fix), with the 4 strongest rows:
  - **Every model levels off at about 2.0 CV lift** → 64% of late orders still missed at the 10% budget → add data the model can't see today: dispatch scans, carrier, holiday and strike calendars.
  - **Scores run about twice too high** on the test period (mean 8.1% predicted vs 4.1% observed) → a fixed probability cut-off would flag the wrong number of orders → flag the top 10% daily; recalibrate on recent data.
  - **Gain over B1 at the 10% budget is within the noise** → the extra features may not change what ops catch → re-check on a fresh period; B1 is a credible lightweight fallback.
  - **Short test window that includes the May 2018 truck drivers' strike**, plus 2,963 undelivered orders with no label → real uncertainty, and the true late rate is likely above 7% → retrain quarterly, monitor monthly, treat stuck "shipped" orders as late in a check.
- Source: README 7.

### 12. Recommendation: score daily, review the top 10%, invest in new data (0:25) ✅
- Answer to the research question in one line: **moderately well. The model catches 36% of late orders by checking 10%, 3.6× random, and the risk comes from promises that are too short for the distance and the destination, not the product.**
- What ops should do: run the score nightly and review the top 10%; use it as a ranking, not a probability; retrain quarterly and track recall monthly.
- The next gain comes from new data (dispatch scans, carrier data, a disruption calendar), not more modelling.
- Source: README 6.

**Total: about 9:20**

---

## Backup slides (for Q&A, not presented)
- B1. Full benchmark table, all 9 models (README 4.1)
- B2. Leakage audit table (README 3.4)
- B3. Why the censoring rule: late orders promised near the data cut-off would never show as delivered (README 1.8, D4)
- B4. Why "late" means a later calendar day: comparing timestamps would inflate the late rate from 6.8% to 8.1% (D3)
- B5. CV fold late rates 2.8% to 18.8%, and why we compare on lift (`reports/figures/02b_cv_fold_late_rates.png`)
- B6. Late rate by customer state (`reports/figures/06_late_rate_by_customer_state.png`)
- B7. Validation-fold confusion matrix at the top 10%: 476 caught, 1,535 missed, 592 false alarms (README 4.3)
- B8. Random split vs time split: PR-AUC 0.189 vs 0.168, recall 27.7% vs 23.7% (README 4.4)
- B9. Ensemble comparison (`reports/figures/tasnim/5_2_ensemble_vs_single.png`, README 5.2)
- B10. `purchase_month` effect by fold (`reports/figures/tasnim/5_1_6_purchase_month_by_fold.png`)
- B11. Test results by month, including the May 2018 truck drivers' strike (README 5.3)
- B12. Calibration: mean predicted 8.1% vs observed 4.1% (`reports/figures/tasnim/5_4_test_calibration.png`)
- B13. Extra features tried in 4.4 and why none were kept (README 4.4)

## Figures

Existing figures use two different styles: Hasan's (blue = on time, orange = late, light background) and Tasnim's (matplotlib defaults). For one consistent deck, restyle the ones we reuse to Hasan's palette.

| # | Figure | Slide | Source | Status |
|---|---|---|---|---|
| F1 | 1–2★ review share, late vs on time | 1 | README 1.1 numbers | to build |
| F2 | Order → score → flag → action flow | 2 | diagram | to build |
| F3 | Monthly volume and late rate | 3 | `reports/figures/02_monthly_volume_and_late_rate.png` | ready |
| F4 | Population funnel | 4 | README 1.8 numbers | to build |
| F5 | Process pipeline with split and gap | 5 | diagram | to build |
| F6 | Distance × promise heatmap | 6 | `reports/figures/05_distance_x_promise_heatmap.png` | ready |
| F7 | Model lift ± std, with B0/B1 reference lines | 7 | README 4.1 numbers | to build |
| F8 | CV lift steps: untuned → tuned → no month | 8 | README 5.1 numbers | to build |
| F9 | Recall at review budget, final vs B1 vs random | 9 | `reports/figures/tasnim/5_3_test_pr_and_budget_curves.png` | restyle |
| F10 | Test permutation importance, top 5–7 | 10 | `reports/figures/outputs/tasnim_5_4_permutation_importance.csv` | restyle |
| F11 | Limitations table | 11 | README 7 | to build |

## Rubric coverage check

| Presentation rubric item | Slides |
|---|---|
| Opens with the problem, why it matters, who it serves | 1, 2, 3 |
| How the project and dataset meet the objective | 4 |
| End-to-end process, concise | 5, 7, 8, 9 |
| Interesting findings along the way | 3, 6, 8, 10 |
| What worked and what did not | 7, 8 (plus the README 6 table in backup if asked) |
| Specific assumptions and limitations, with impact and fix | 11 |
| Clear storyline, labelled visuals, memorable takeaways | headlines throughout; 6, 9, 12 |
| Fits 10 minutes | about 9:20 planned; two timed rehearsals |
