# Team conventions

Read this before starting your work. You can work locally and upload through GitHub's website, or use Git if you prefer. These rules help us keep everyone's work consistent when combining it into the final notebook.

## 1. Setup (once)

You don't need to know Git to contribute.

1. Open the GitHub repository and click **Code → Download ZIP**.
2. Extract the ZIP on your computer.
3. Open the project folder in VS Code, JupyterLab, or your preferred editor.
4. Set up Python 3.12 and install the dependencies using the instructions for your operating system.

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` instead of `python` for the remaining commands.

All 8 tests should pass. If anything fails, share the error in the group chat.

If you're already familiar with Git, you can clone the repository instead of downloading the ZIP. Both approaches are fine.

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

Nobody else edits the final notebook. Everyone works on their assigned section locally and uploads their notebook to wip/. If your work includes reusable functions in src/, upload those too. Jay will review and integrate everyone's code, graphs, and explanations into the final notebook and README.

## 3. Sharing your work

### Option 1: Work locally and upload (recommended)

No Git knowledge required.

1. Download the repository ZIP and work on your assigned section locally.
2. Save your notebook inside `wip/`, using your name (e.g., `wip/hasan_eda.ipynb`).
3. When ready, open the `wip/` folder on GitHub.
4. Click **Add file → Upload files**, select your notebook, and commit the upload.
5. Jay will review the contributions and integrate them into the final notebook and README.

Upload updated versions periodically so GitHub records your progress, not just your final submission.

If your work includes changes to `src/` or other shared files, upload those files in their original folders too, and inform Jay so nothing gets overwritten. Do not overwrite another teammate's work.

### Option 2: Use Git (optional)

If you're familiar with Git or want to learn:

- Clone the repository.
- Create a branch for your work, e.g., `mia/seller-history`.
- Commit your changes with descriptive messages.
- Run the tests before submitting.
- Open a pull request into `main` for Jay to review and merge.

### Rules for everyone

- Use the shared functions in `src/` and follow the coding and leakage rules below.
- Keep all code, graphs, results, and explanations reproducible.
- Make sure your notebook runs from top to bottom.
- Don't edit the final notebook or another teammate's files.
- Don't evaluate models on the test set during development. Only Tasnim runs the final evaluation, once the model has been selected.
- Never upload `.venv/`, `mlruns/`, or files excluded by `.gitignore`.

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
