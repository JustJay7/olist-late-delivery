# Slide outline: Predicting late deliveries on Olist

Owner: Peace. Draft v2, 23 Oct 2026 (v2 adds Jingyu's 4.2–4.3 findings). Everything here is still open to change; comments welcome in the group chat.

**Time limit:** 10 minutes. **Target:** about 9:15 of content, which leaves room for speaker handovers. 12 slides plus backup.

**Storyline in one sentence:** late orders destroy customer satisfaction, Olist's padded promises don't prevent them, and a model scored at approval lets ops catch about twice as many late orders as checking at random.

**Rules for the deck**
- Every slide headline is the takeaway, written as a full sentence, not a topic label.
- At most one main visual per slide, with a title, labelled axes and units.
- Every number must come from the final notebook (CONVENTIONS §6). Numbers marked *provisional* will change.

**Status key:** ✅ content final · 🟡 provisional (numbers will change) · ⏳ waiting on a teammate

---

## Slide-by-slide

### 0. Title (0:10) ✅
- Predicting Late Deliveries at Approval Time: Olist Brazilian E-Commerce
- Team names, INSY 674, Fall 2026

### 1. Late orders turn happy customers into 1-star reviews (1:00) ✅
*Rubric: Objective (the problem and why it matters)*
- 1 in 14 delivered orders arrived late (5,986 of 85,371, 7.0%).
- Late orders average **2.21★ vs 4.28★** on time, and **62.5%** get a 1–2★ review (vs 9.3% on time).
- **Visual:** side-by-side bars of the share of 1–2★ reviews, on time vs late.
- *Note to say aloud:* reviews only motivate the problem; they are never used as model inputs.
- Source: README 1.1, notebook 1.2.

### 2. Ops needs a daily shortlist of risky orders, not a better average (0:50) ✅
*Rubric: Objective (who it serves and how it's used)*
- User: Olist's operations team. Each morning, yesterday's approved orders get a risk score and the riskiest 10% are flagged.
- Actions: push the seller to ship, chase the carrier, warn the customer with a new date.
- The model ranks orders for people to review; it doesn't change dates by itself.
- **Visual:** simple flow diagram: order approved → nightly risk score → top 10% flagged → ops action.
- Source: README 1.3.

### 3. Padding the promise isn't enough (0:50) ✅
*Rubric: Objective (current workaround) and an interesting finding*
- The median promise is **24 days**, but the median delivery takes **10**. Every order gets the same padding.
- Late rate still spiked: **12.4%** (Nov 2017; likely Black Friday volume, still to confirm with Hasan's EDA), **14.1%** (Feb 2018), **19.0%** (Mar 2018).
- **Visual:** monthly late-rate line chart, Jan 2017 to Jul 2018, spikes annotated, train/test split marked.
- Source: README 1.4, notebook 1.3.

### 4. Olist's public data lets us predict at the moment ops can act (0:50) ✅
*Rubric: Objective (how the project and dataset meet it)*
- Dataset: Olist Kaggle release, 9 joined tables (orders, items, products, sellers, customers, geolocation).
- Framing: supervised binary classification (late or not), scored offline in a daily batch, **predicting at order approval** using only information available then.
- Funnel: 99,441 orders → **85,371** usable (delivered, dated, 2017 onward, censoring rule).
- Metric: **PR-AUC**, plus **recall in the top 10%** ("of all late orders, how many does ops catch?"). Accuracy is useless here: predicting "on time" for everything scores 95.9% and catches zero late orders.
- **Visual:** population funnel bar (left) plus one-line metric definition (right).
- Source: README 1.5–1.8, DECISIONS D2–D8.

### 5. Our process: the checklist from end to end, with no peeking at the future (0:30) ✅
*Rubric: Process (end-to-end, concise)*
- **Visual only:** a horizontal pipeline of Frame → Explore → Prepare → Compare models → Tune and ensemble → Test once.
- Annotate the time split under it: train Jan 2017–Mar 2018 (64,038 orders) | test Apr–Jul 2018 (21,333 orders).
- Say: "We split by date because the model will score future orders."

### 6. What the data told us (1:00) ⏳ Hasan
*Rubric: Process and Insights (interesting findings)*
- 2–3 strongest EDA findings, each as a one-line takeaway with a plot. Candidates to check with Hasan:
  - late rate by customer state (map or bar)
  - promised days vs late rate
  - distance vs late rate
- Already available as a finding: the late rate halves from **8.0%** (train) to **4.1%** (test), which matters later (drift).
- Source: README §2 (not yet written).

### 7. We built one leak-free pipeline that every model shares (0:40) 🟡 Mia
*Rubric: Process*
- 22 approval-time features → 20 after dropping zip code (13,584 values) and approval month (the same as purchase month 98.2% of the time).
- Median or "unknown" filling plus missing-value flags; log transform on 13 skewed columns; scaling; one-hot encoding → 103 inputs.
- Preprocessing is fitted inside each CV fold only; banned post-approval columns are listed in config.
- **Visual:** compact pipeline diagram with small icons; full leakage audit table goes in backup.
- Source: `wip/mia_preprocessing.ipynb`, README §3.

### 8. Simple models beat complex ones, and a 2-feature baseline is hard to beat (1:00) 🟡 Fidan
*Rubric: Process and Insights (what worked and what didn't)*
- 7 models from 5 families, compared with 5-fold time-series CV.
- Current numbers: logistic regression **0.158** PR-AUC vs the 2-feature baseline B1 **0.150**; random forest (0.141) and gradient boosting (0.139) did worse.
- Fold scores swing a lot (std ±0.12) because fold late rates range from 2.8% to 18.8%, the same drift seen on slide 6.
- **Visual:** horizontal bar chart of mean PR-AUC ± std per model, with B0 and B1 drawn as reference lines.
- **What didn't work (error analysis, Jingyu):** the model is essentially a long cross-state route detector. The late orders it catches travel about 995 km and are almost all cross-state; the ones it misses are shorter (about 730 km) and more often within one state. Its false alarms are long, expensive orders that arrived on time.
- Switching models doesn't fix it: logistic regression and random forest miss the **same 68%** of late orders. The limit is the data, not the algorithm (sets up slide 11).
- Numbers will change once re-run through `src/evaluation.py`.
- Source: `wip/Models.ipynb`, `wip/jingyu_errors.ipynb` 4.3, README 4.1/4.3/4.5.

### 9. Where the customer lives matters most, about twice as much as the promised window (1:00) 🟡 Jingyu
*Rubric: Insights (memorable takeaway)*
- Permutation importance (drop in PR-AUC when a feature is shuffled): **customer state 0.112**, promised days 0.052, same-state share 0.032, distance 0.021, seller state 0.018; everything else < 0.01.
- Takeaway: **geography and the promised window drive risk; order size, price and product barely matter.**
- Direction (logistic regression coefficients): customers in AL and RJ are riskier, PR and MG safer; longer promised windows lower the risk.
- **Visual:** Jingyu's top-10 permutation importance bar chart (re-styled, with geography features highlighted).
- Caveat to say aloud: these are associations, not causes, and they come from one validation fold (Mar 2018, an 18.8% late-rate spike).
- Source: `wip/jingyu_errors.ipynb` 4.2.3, README 4.2.

### 10. The final model catches X% of late orders by checking 10% (1:00) ⏳ Tasnim
*Rubric: Process (fine-tuning) and the objective payoff*
- Final model, its search method (randomized search), and whether the ensemble helped.
- **Test-set result vs B0 and B1:** PR-AUC, recall at 5/10/20%, lift over the base rate.
- **Visual:** recall-at-budget chart (x = % of orders checked, y = % of late orders caught) for final, B1 and random.
- Source: README 5.1–5.4.

### 11. Assumptions and limitations, and how we'd fix them (1:00) ✅ (model limits ⏳ Jay)
*Rubric: Assumptions and Limitations (specific, with impact and fix)*
- **Visual:** 3-column table (Limitation | Impact | Fix), with the 4 strongest rows:
  - **Drift:** the late rate halves from train (8.0%) to test (4.1%) → the cut-off may over-flag → recalibrate and retrain monthly.
  - **No label for 2,963 undelivered orders** → the true late rate is likely higher than 7% → treat orders still "shipped" past their promise as late.
  - **No carrier, holiday or weather data** → main causes of delay can't be learned; this is why short within-state late orders are missed (slide 8) → add the Brazilian holiday calendar, request carrier data.
  - **The 10% ops capacity is a guess** → we also report 5% and 20%.
- Add 1 model-level limitation once section 5 is done.
- Source: README §7.

### 12. Recommendation (0:25) ⏳
- One sentence answering both parts of the research question, with numbers from slide 10 and slide 9.
- What ops should do on Monday: run the score nightly, review the top 10%, track recall monthly.
- Source: README §6 (Jay).

**Total: about 9:15**

---

## Backup slides (for Q&A, not presented)
- B1. Full benchmark table (mean ± std for all metrics)
- B2. Leakage audit table (Mia 3.4b)
- B3. Why the censoring rule: late orders promised near the data cut-off would never show as delivered
- B4. Why "late" means a later calendar day: comparing timestamps would inflate the late rate from 6.8% to 8.1%
- B5. Fold-by-fold PR-AUC vs fold late rate
- B6. Top-10% confusion matrix on the validation fold: 1,068 flagged, 485 truly late (45% precision vs 18.8% base rate, 2.4× lift), 1,526 late orders missed (Jingyu 4.3.1)
- B7. Error overlap, logistic regression vs random forest: 1,366 missed by both, 160 / 182 caught by only one, 303 caught by both (Jingyu 4.3.4)
- B8. Top-15 logistic regression coefficients (Jingyu 4.2.2)

## Figures needed (save to `reports/figures/`)

| # | Figure | Slide | From | Status |
|---|---|---|---|---|
| F1 | Review stars, late vs on time | 1 | Peace (notebook 1.2 data) | can build now |
| F2 | Order → score → flag → action flow | 2 | Peace | can build now |
| F3 | Monthly late rate with spikes and split marked | 3 | Peace (notebook 1.3 data) | can build now |
| F4 | Population funnel | 4 | Peace (notebook 1.1 data) | can build now |
| F5 | Process pipeline graphic | 5 | Peace | can build now |
| F6 | 2–3 EDA plots | 6 | Hasan | waiting |
| F7 | Preprocessing pipeline diagram | 7 | Peace, from Mia's notebook | can build now |
| F8 | Model comparison bars ± std | 8 | Fidan | waiting on final numbers |
| F8b | Caught vs missed late orders: distance and same-state share (TP vs FN) | 8 | Peace, from Jingyu 4.3.2 numbers | can build now |
| F9 | Permutation importance, top 10 | 9 | Jingyu (plot exists in notebook; needs saving to `reports/figures/` or restyling) | available |
| F10 | Recall-at-budget curve, final vs baselines | 10 | Tasnim | waiting |

## Rubric coverage check

| Presentation rubric item | Slides |
|---|---|
| Opens with the problem, why it matters, who it serves | 1, 2, 3 |
| How the project and dataset meet the objective | 4 |
| End-to-end process, concise | 5, 7, 8, 10 |
| Interesting findings along the way | 3, 6, 9 |
| What worked and what did not | 8, 10 |
| Specific assumptions and limitations, with impact and fix | 11 |
| Clear storyline, labelled visuals, memorable takeaways | headlines throughout, 9, 12 |
| Fits 10 minutes | about 9:15 planned; two timed rehearsals |


