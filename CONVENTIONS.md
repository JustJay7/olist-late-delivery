# Team conventions

Read this before your first commit. These rules exist so the final notebook can be assembled
in one day without rework.

## 1. Setup (once)

```bash
git clone <repo-url>
cd olist-late-delivery
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                          # must print "8 passed"
```

## 2. Where your work goes

| What | Where | Owner |
|---|---|---|
| Locked decisions, target, population, split | `DECISIONS.md`, `src/config.py`, `src/data.py` | Jay |
| Shared metrics and CV strategy | `src/evaluation.py` | Jay |
| EDA and profiling | `wip/hasan_eda.ipynb` | Hasan |
| Cleaning, features, preprocessing pipeline | `src/features.py`, `src/preprocessing.py` | Mia |
| Baselines and benchmark | `src/models.py`, `wip/fidan_benchmark.ipynb` | Fidan |
| Importance and error analysis | `wip/jingyu_errors.ipynb` | Jingyu |
| Tuning, ensembles, final test evaluation | `wip/tasnim_tuning.ipynb` | Tasnim |
| Slides | `slides/` | Peace |
| Figures for README and slides | `reports/figures/<section>_<name>.png` | Anyone |
| Final graded notebook and README | `notebooks/olist_late_delivery.ipynb`, `README.md` | Jay |

Nobody else edits the final notebook. Notebooks do not merge cleanly in Git, so you hand over
functions in `src/` plus a short note of which cells to add, and Jay integrates them.

## 3. Git

- One branch per person per task: `<name>/<topic>` (e.g. `mia/seller-history`).
- Open a pull request into `main`; Jay reviews and merges.
- Run `pytest -q` before opening a pull request.
- Commit messages say what changed: `Add point-in-time seller late rate`, not `update`.
- Never commit `mlruns/`, `.venv/`, or anything in `.gitignore`.

## 4. Code rules

- Import every path, date, seed and threshold from `src/config.py`. No hard-coded values.
- Start from `build_population()` and `time_split()` in `src/data.py`. Do not re-filter orders yourself.
- Comments explain **why**, not what: `# median, not mean: freight is heavily right-skewed`.
- Fail loudly. Raise an error on unexpected input; no bare `except:` and no silent fallbacks.
- Use `config.RANDOM_STATE` for every seed.
- Functions take a DataFrame and return a DataFrame or Series, so they drop into a pipeline.

## 5. Leakage rules (non-negotiable)

1. Never use a column in `config.BANNED_FEATURE_COLUMNS`, or anything computed from one.
2. History features (seller late rate, route late rate) use only orders delivered **before** the
   current order's purchase time. Use `pd.merge_asof` on the sorted timeline.
3. Every fitted step (imputer, encoder, scaler, resampler, feature selector) goes inside the
   sklearn or imblearn `Pipeline`, so it is fitted on training folds only.
4. Use `TimeSeriesSplit(n_splits=config.N_CV_SPLITS)` for CV. Never shuffle.
5. **Nobody evaluates on the test set** until the final model is chosen. Tasnim runs it once,
   after model selection is frozen, and hands the result to Jay for the README.

## 6. Plots

Every plot has a title, labelled axes with units, and a legend if more than one series.
Save it to `reports/figures/`. The markdown cell after it states the takeaway in one sentence,
e.g. *"Orders bought in February and March 2018 were late 3 to 4 times more often than in 2017."*

## 7. Numbers

Any number in the README or slides must come from a cell output in the final notebook.
If a number changes, update it everywhere in the same commit.

## 8. Experiment tracking (MLflow)

- Experiment name: `olist-late-delivery`
- Run name: `<model>_<variant>` (e.g. `lgbm_classweight`)
- Log: all parameters, CV mean and std of every score in `src.evaluation.scorers()`, and fit time.

## 8a. Scoring every model the same way

```python
from sklearn.model_selection import cross_validate
from src.evaluation import cv_splitter, scorers, summarize_cv

res = cross_validate(pipeline, X_train, y_train, cv=cv_splitter(), scoring=scorers())
summarize_cv(res, "lgbm_default")
```

Use `refit="pr_auc"` in `RandomizedSearchCV`. Never write your own metric or CV splitter.

## 9. Freeze points

| Day | What freezes |
|---|---|
| 1 | Decisions D1 to D12 |
| 4 | Feature set and shortlist of 3 models |
| 5 | Final model and every reported number |
