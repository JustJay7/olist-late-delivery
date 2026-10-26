# Slide outline: Predicting late deliveries on Olist

Owner: Peace. v4, 26 Oct 2026. Matches the deck `slides/olist_late_delivery_presentation.pptx` (built on the team's Canva template), after a review against the presentation rubric. Comments welcome in the group chat.

**Time limit:** 10 minutes. **Planned:** 9:10 of content across 13 presented slides, leaving about 50 seconds for speaker handovers. Timings and a script are in each slide's speaker notes, with a "Speaker: ____" line to fill in.

**Storyline in one sentence:** late orders come with far worse reviews, Olist's padded promises don't prevent them, and a simple model scored at approval catches 36% of late orders by checking just 10% of orders (3.6× random checks). The next gains need new data, not more modelling.

**Rules for the deck**
- Every slide headline is the takeaway, written as a full sentence, not a topic label.
- At most one main visual per slide, with a title, labels and units. Charts and tables are native PowerPoint objects, so numbers can be edited.
- Every number comes from the final README / notebooks (CONVENTIONS §6). Section references are in the speaker notes.
- Fonts: Arial Black (headings) and Calibri (body), because the template's Canva fonts aren't installed on most machines.

---

## Slide-by-slide

### 1. Title (0:10)
- Predicting late deliveries: flagging risky Olist orders at approval, so ops can act before the customer notices
- Team names, INSY 674, Fall 2026

### 2. Late orders come with far worse reviews (0:45)
*Rubric: Objective (the problem and why it matters)*
- 1 in 14 delivered orders arrived late (5,986 of 85,371).
- Late orders average **2.21★ vs 4.28★**, and **62.5%** get a 1–2★ review (vs 9.3% on time): 6.7× more likely.
- **Visual:** column chart of the 1–2★ share, on time vs late.
- Reviews only size the problem (an association); they are never model inputs.
- Source: README 1.1.

### 3. Ops needs a daily shortlist of risky orders (0:40)
*Rubric: Objective (who it serves and how it's used)*
- Order approved → nightly risk score → top 10% flagged (about 18 a day) → ops acts (push the seller, chase the carrier, warn the customer).
- The model ranks orders for people to review; it never changes dates by itself.
- **Visual:** 4-step flow with template icons.
- Source: README 1.3, 5.4.

### 4. Padding the promise isn't enough: lateness comes in waves (0:45)
*Rubric: Objective (current workaround) and an interesting finding*
- Median promise **24 days** vs median delivery **10**: a one-size safeguard, with no order-level risk score.
- Spikes: **12.4%** (Nov 2017), **14.1%** (Feb 2018), **19.0%** (Mar 2018). Those three months hold **33% of training orders but 62% of the late ones**. Even São Paulo's late rate nearly tripled (3.4% → 9.4%).
- **Visual:** monthly late-rate line chart, training and test periods in different colours.
- Source: README 1.4, 2.2.

### 5. Olist's public data lets us predict at the moment ops can act (0:45)
*Rubric: Objective (how the project and dataset meet it)*
- Funnel: 99,441 orders → **85,371** usable (delivered, dated, 2017 onward, censoring rule).
- Framing: supervised binary classification, daily batch, predicting at approval with only what is known then.
- Metrics: **PR-AUC** and **recall in the top 10%**. Accuracy is useless: "always on time" is 95.9% accurate and catches nothing.
- **Visual:** funnel bars plus three cards (frame, metrics, why not accuracy).
- Source: README 1.5–1.8.

### 6. Our process, end to end, with no peeking at the future (0:35)
*Rubric: Process (end-to-end, concise)*
- Six stages with owners: Frame (Jay), Explore (Hasan), Prepare (Mia), Compare 7 models (Fidan, Jingyu), Tune + ensembles (Tasnim), Test once (Tasnim).
- Time split: train Jan 2017–Mar 2018 (64,038 orders, 8.0% late) | 2,000-order gap | test Apr–Jul 2018 (21,333, 4.1% late).
- One leak-free pipeline refitted inside every fold (103 inputs); 5-fold time-series CV with the same gap; a random split would have flattered us (27.7% vs the honest 23.7% recall).
- Source: README 3, 4.4, DECISIONS D6–D7.

### 7. Late deliveries happen when the promise is too short for the distance (0:50)
*Rubric: Process and Insights (interesting findings)*
- Over 889 km with a promise of 22 days or less: **about 1 in 5 late**. Under 137 km with 30+ days: **1.7%**.
- Where it goes > what's in it: Rio 15.6% late, 2.9× Minas Gerais (5.3%); product categories only span 5.9–10.2%.
- **Visual:** distance × promise heatmap as a native table.
- Source: README 2.3, 2.7.

### 8. Simple beats complex: logistic regression led from the start (0:45)
*Rubric: Process and Insights (what worked and what didn't)*
- 7 models from 5 families plus 2 baselines, compared on **lift** (PR-AUC ÷ the fold's late rate), shown as **mean ± std over 5 folds**: LR 1.81 ± 0.21, B1 1.76 ± 0.35, random forest 1.61 ± 0.16, gradient boosting 1.54 ± 0.24, random 1.00 ± 0.03.
- Trees overfit (training lift 6–12× validation). The EDA pointed to trees; they didn't win.
- **Shortlist:** LR, random forest, gradient boosting, one per family so their errors differ.
- Source: README 4.1, 4.5, 2.7.

### 9. Dropping one misleading column helped more than all the tuning (0:45)
*Rubric: Process (fine-tuning) and Insights*
- Randomized search, 90 trials (5–7 hyperparameters, several continuous). CV lift, mean ± std: LR 1.81 ± 0.21 → tuned 1.87 ± 0.21 → tuned without `purchase_month` 1.98 ± 0.24; random forest 1.80 ± 0.30 and gradient boosting 1.79 ± 0.28 (both tuned, no month).
- Caveat on the slide: gains sit within the fold noise, but dropping the month helped all three models.
- Why: with 15 months of data the month column remembered which months had been bad (February 3.0% late in 2017, 14.1% in 2018).
- No gain from 4 voting ensembles or 8 extra features. Final model: tuned logistic regression (L1).
- Source: README 4.4, 5.1, 5.2.

### 10. Checking the riskiest 10% of orders catches 36% of late deliveries (0:50)
*Rubric: Fine-tuning result and the objective payoff*
- Test set (Apr–Jul 2018), touched once: **316 of 876** late orders caught, vs 294 (B1) and about 88 (random). Flagged orders were late **3.6×** as often.
- **Visual:** recall at 5% / 10% / 20% review budgets for the final model, B1 and random.
- Honest caveat: the PR-AUC gain over B1 is real (95% CI +0.006 to +0.027); at the 10% budget it is within the noise (+2.5 points, CI −0.4 to +4.8).
- Source: README 5.3, 5.4, 6.

### 11. Promise and distance drive the risk; "ordinary" late orders stay invisible (0:50)
*Rubric: Insights and error analysis*
- Test-set permutation importance: promised days, distance, customer state, freight, same-state share. A promise 1 SD longer roughly halves the odds of being late (×0.52).
- Error analysis (last CV fold, Feb–Mar 2018, before tuning): missed late orders look ordinary, shorter than the ones caught (727 vs 1,011 km). All 3 shortlisted models miss the same 66%.
- What a person would have needed: the carrier hand-off, carrier backlogs, strikes and holidays, the network-wide delay.
- Source: README 4.3, 5.4.

### 12. Assumptions and limitations, and how we'd fix them (0:50)
*Rubric: Assumptions and Limitations (specific, with impact and fix)*
- Assumptions: the promise is the date shown at checkout; ops can review about 10% a day (5% and 20% also reported); 2017–Mar 2018 patterns hold in Apr–Jul 2018.
- Limitation | impact | fix table: the ~2.0 lift ceiling; scores about 2× too high on the test period; gain over B1 within the noise at 10%; a short test window with the May 2018 strike and 2,963 orders never delivered.
- Source: README 7.

### 13. Score daily, review the top 10%, invest in new data (0:40)
*Rubric: Conclusion, and what worked / what didn't*
- Answer: moderately well (36% of late orders caught by checking 10%, 3.6× random); the risk comes from a promise too short for the distance and from the destination, not the product.
- What worked: the time-based split and folds, comparing on lift, dropping `purchase_month`, a regularised logistic regression. What didn't: trees and ensembles, extra features, native categorical splits.
- Actions: score nightly and review the top 10%; track recall monthly and retrain quarterly; invest in new data.
- Source: README 6.

### 14. Thank you · Questions?
- Repo link and the contributor list from the GitHub repo (names, handles and roles).

**Total presented: 9:10**

---

## Backup slides (in the deck, for Q&A)
- Full model comparison table, all 9 models (README 4.1)
- How we kept the future out of the model: leakage controls, censoring rule, calendar-day target (README 3.4, DECISIONS)
- Late rate by customer state (Hasan's chart, README 2.3)
- Tuning and ensembles: what we tried, and the selection rule (README 5.1–5.2)
- Test results by month, including the May 2018 strike, and why scores are a ranking (README 5.3–5.4)

## Rubric coverage check

| Presentation rubric item | Slides |
|---|---|
| Opens with the problem, why it matters, who it serves | 2, 3, 4 |
| How the project and dataset meet the objective | 5 |
| End-to-end process, concise | 6, 8, 9, 10 |
| Interesting findings along the way | 4, 7, 9, 11 |
| What worked and what did not | 8, 9, 13 |
| Specific assumptions and limitations, with impact and fix | 12 |
| Clear storyline, labelled visuals, memorable takeaways | headlines throughout; 7, 10, 13 |
| Fits 10 minutes | 9:10 planned; assign speakers (5–6 keeps handovers short) and run two timed rehearsals |
