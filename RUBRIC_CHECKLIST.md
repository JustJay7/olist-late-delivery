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
| GitHub repo link to project | Repo `README.md` | Jay | draft |
| Comments explaining Frame, Explore, Prepare, Measure and compare, Fine-Tune | Notebook section headings + `README.md` sections 1 to 5 | Jay | draft |
| Slides | `slides/` | Peace | todo |
| Notebook `.ipynb` + `.html` with results, all member names (syllabus) | `notebooks/` | Jay | todo |
| Raw dataset as CSV (syllabus) | `data/raw/` | Jay | done |
| Reproducible from raw data + notebook (syllabus) | Fresh clone test | Jay | todo |

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
| Profiles every attribute: type, % missing, noise, distribution, usefulness | README 2.1 profile table | Hasan | todo |
| Identifies the target | README 2.2 (target defined by Jay in D3; Hasan profiles it) | Hasan | todo |
| Labelled plots, each with a takeaway | README 2.3 + notebook | Hasan | todo |
| Correlations, each with a takeaway | README 2.4 | Hasan | todo |
| Lists promising transformations | README 2.5 | Hasan | todo |
| Lists extra data that would help | README 2.6 | Hasan | todo |
| Summarises learnings | README 2.7 | Hasan | todo |

### Prepare the Data, Measure and Compare Performance

| Excellent band item | Where | Owner | Status |
|---|---|---|---|
| Cleaning justified | README 3.1 | Mia | todo |
| Feature engineering justified | README 3.2 | Mia | todo |
| Scaling justified | README 3.3 | Mia | todo |
| Automated in a leak-free pipeline, separated train/test | README 3.4 + leakage audit table | Mia | todo |
| Models from several families | README 4.1 | Fidan | todo |
| N-fold CV, mean and std | README 4.1 table (`src/evaluation.py`) | Fidan | todo |
| Analyses key variables | README 4.2 | Jingyu | todo |
| Analyses error types | README 4.3 | Jingyu | todo |
| Iterates (feature selection / engineering round) | README 4.4 | Mia, using Jingyu's findings | todo |
| Shortlists 3 to 5 models with reasons | README 4.5 | Fidan | todo |

### Fine-Tune the System

| Excellent band item | Where | Owner | Status |
|---|---|---|---|
| Tunes shortlisted models with CV | README 5.1 | Tasnim | todo |
| Justified search method | README 5.1 | Tasnim | todo |
| Tries ensembles | README 5.2 | Tasnim | todo |
| Evaluates on the test set (once) | README 5.3 | Tasnim | todo |
| Explains the final model | README 5.4 | Tasnim, Jingyu | todo |
| Performance vs baseline | README 5.3 (B1 from Fidan) | Tasnim | todo |
| Limitations | README 7 | Jay | draft |

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
