# Rubric coverage tracker

Every line of the "Excellent" band, in the rubric's own wording, mapped to where it is met and who
owns it. The project lead updates Status during integration. Nothing is submitted with an open line.

Owners: **Jay** (lead, framing, integration), **Hasan** (EDA and profiling), **Mia** (cleaning,
features, preprocessing pipeline), **Fidan** (baselines and benchmarking), **Jingyu** (interpretation
and error analysis), **Tasnim** (tuning, ensembles, final test evaluation), **Peace** (slides).

Status: `todo` / `draft` / `done`

## Submission requirements

| Requirement | Where | Owner | Status |
|---|---|---|---|
| GitHub repo link to project | Repo `README.md` | Jay | done |
| Comments explaining Frame, Explore, Prepare, Measure and compare, Fine-Tune | `README.md` sections 1 to 5, plus one notebook per section in `notebooks/` | Jay | done |
| Slides | `slides/` | Peace | todo |
| Notebooks with results, all member names | `notebooks/` (05 also as `.html`); names and roles in `README.md` | Jay | done |
| Raw dataset as CSV (syllabus) | `data/raw/` | Jay | done |
| Reproducible from raw data + notebooks | Fresh clone: notebooks 01 to 04c re-run and match; 05 paths checked (full run takes about 1.5 hours) | Jay | done |
| Student IDs for every member | `README.md` header | Jay | todo |

## Source code rubric

### Frame the Problem and Look at the Big Picture

| Excellent band item | Where | Owner | Status |
|---|---|---|---|
| States the objective (business goal) | README 1.1 | Jay | done |
| Who uses the solution and how | README 1.3 | Jay | done |
| Justifies the framing (supervised, task type, online/offline) | README 1.5 | Jay | done |
| Current solutions / workarounds (checklist) | README 1.4 | Jay | done |
| Performance measure and its alignment with the objective (checklist) | README 1.6 | Jay | done |

### Explore the Data

| Excellent band item | Where | Owner | Status |
|---|---|---|---|
| Profiles every attribute: type, % missing, noise, distribution, usefulness | README 2.1 profile table | Hasan | done |
| Identifies the target | README 2.2 (target defined by Jay in D3; Hasan profiles it) | Hasan | done |
| Labelled plots, each with a takeaway | README 2.3 + notebook | Hasan | done |
| Correlations, each with a takeaway | README 2.4 | Hasan | done |
| Lists promising transformations | README 2.5 | Hasan | done |
| Lists extra data that would help | README 2.6 | Hasan | done |
| Summarises learnings | README 2.7 | Hasan | done |

### Prepare the Data, Measure and Compare Performance

| Excellent band item | Where | Owner | Status |
|---|---|---|---|
| Cleaning justified | README 3.1 | Mia | done |
| Feature engineering justified | README 3.2 | Mia | done |
| Scaling justified | README 3.3 | Mia | done |
| Automated in a leak-free pipeline, separated train/test | README 3.4 + leakage audit table | Mia | done |
| Models from several families | README 4.1 | Fidan | done |
| N-fold CV, mean and std | README 4.1 table (`src/evaluation.py`) | Fidan | done |
| Analyses key variables | README 4.2 | Jingyu | done |
| Analyses error types | README 4.3 | Jingyu | done |
| Iterates (feature selection / engineering round) | README 4.4, `04c_feature_iteration.ipynb` | Jay, using Jingyu's findings | done |
| Shortlists 3 to 5 models with reasons | README 4.5 | Fidan | done |

### Fine-Tune the System

| Excellent band item | Where | Owner | Status |
|---|---|---|---|
| Tunes shortlisted models with CV | README 5.1 | Tasnim | done |
| Justified search method | README 5.1 | Tasnim | done |
| Tries ensembles | README 5.2 | Tasnim | done |
| Evaluates on the test set (once) | README 5.3 | Tasnim | done |
| Explains the final model | README 5.4 | Tasnim, Jingyu | done |
| Performance vs baseline | README 5.3 (B1 from Fidan) | Tasnim | done |
| Limitations | README 7 | Jay | done |

## Presentation rubric

| Excellent band item | Where | Owner | Status |
|---|---|---|---|
| Opens with the problem, why it matters, who it serves | Slides 1 to 2 (content from README 1) | Peace | todo |
| Explains how the project and dataset meet the objective | Slide 3 | Peace | todo |
| End-to-end process, concise | Slides 4 to 8 | Peace | todo |
| Interesting findings noticed along the way | Slides 5, 9 | Peace | todo |
| What worked and what did not, with explanations | Slide 7 | Peace | todo |
| Specific assumptions and limitations, with impact and fix | Slide 10 (content from README 7) | Peace, Jay | todo |
| Clear storyline, labelled visuals, memorable takeaways | Whole deck | Peace | todo |
| Fits the time limit (10 min) | Two timed rehearsals | Peace | todo |
