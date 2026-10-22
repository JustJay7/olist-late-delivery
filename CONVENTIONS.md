# Team conventions

This is just a quick guide to keep everyone's work consistent. **You don't need to know Git or GitHub to contribute.** Download the project, work on your assigned section locally, and upload your notebook when you're done. I'll handle combining everything into the final submission.

## 1. Getting started

1. Open the GitHub repository and click **Code → Download ZIP**.
2. Extract the ZIP and open the folder in VS Code, JupyterLab, or your preferred editor.
3. Set up Python 3.12 and install the required packages.

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` instead.

Once installed, run `python -m pytest -q` to check the setup. All 8 tests should pass.

If anything doesn't work, just ask in the group chat.

## 2. Who's working on what

| Person | Responsibility | Where to save |
|---|---|---|
| Jay | Shared dataset, evaluation code, README, final integration | `src/`, `notebooks/`, `README.md` |
| Hasan | Exploratory data analysis | `wip/hasan_eda.ipynb` |
| Mia | Cleaning, feature engineering, preprocessing | `wip/mia_preprocessing.ipynb` |
| Fidan | Baselines and model comparison | `wip/fidan_benchmark.ipynb` |
| Jingyu | Feature importance and error analysis | `wip/jingyu_errors.ipynb` |
| Tasnim | Tuning, ensembles, final evaluation | `wip/tasnim_tuning.ipynb` |
| Peace | Presentation | `slides/` |

Everyone can work in their own notebook. There's no need to write separate Python modules unless you want to. If you've created reusable functions in `src/`, share those too.

Please don't edit `notebooks/olist_late_delivery.ipynb` directly. That's where I'll combine everyone's work.

## 3. How to share your work

**Option 1: Upload through GitHub (easiest)**

1. Complete your section locally and save your notebook in `wip/`.
2. Open the repository on GitHub and navigate to `wip/`.
3. Click **Add file → Upload files** and select your notebook.
4. Click **Commit changes** to save it.

That's it! I'll take care of integrating everything.

You can upload updated versions as you make progress, so your contributions are recorded on GitHub. If you change shared files or add new dependencies, just let me know.

**Option 2: Use Git (optional)**

If you're comfortable with Git or want to learn, you can clone the repository, work on your own branch, and submit a pull request instead. Both approaches are completely fine.

## 4. A few things to keep consistent

These are the main things to remember while working:

- **Use the shared dataset and code.** The `src/` folder already contains functions for loading data, defining late deliveries, splitting the dataset, and evaluating models. Use these rather than creating separate versions so everyone's results match.
- **Keep the same train/test split.** Don't independently filter or reshuffle orders. Use `build_population()` and `time_split()` from `src/data.py`.
- **Avoid data leakage.** We're predicting late deliveries using information available when an order is approved. Don't use actual delivery dates, review scores, or other information from after approval as model inputs. Check `config.BANNED_FEATURE_COLUMNS` for excluded columns.
- **Don't use the test set during development.** Train and compare models using the training data and time-based cross-validation. Tasnim will run the final test evaluation once we've selected the model.
- **Make your work reproducible.** Include your code, results, graphs, and explanations in your notebook, and check that it runs from top to bottom.

For features based on historical seller or delivery performance, only use information that would already have been available at the time of prediction. Any preprocessing that learns from the data, such as scaling, encoding, imputation, or resampling, should be fitted within the training folds, not on the full dataset.

## 5. Models and evaluation

*Only relevant if you're training or evaluating models.*

To keep model comparisons fair, please use the shared functions in `src/evaluation.py` instead of creating different metrics or cross-validation methods.

Here's the common evaluation code:

```python
from sklearn.model_selection import cross_validate
from src.evaluation import cv_splitter, scorers, summarize_cv

res = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv_splitter(),
    scoring=scorers()
)
summarize_cv(res, "model_name")
```

Our main metric is **PR-AUC**, and we also care about how many late deliveries a model identifies among the highest-risk 10% of orders.

For hyperparameter tuning, use PR-AUC as the selection metric (`refit="pr_auc"`).

MLflow is available for tracking model experiments if you're using it, but you don't need to set it up just to contribute a notebook. At minimum, record the model settings and evaluation results so they can be reproduced in the final notebook.

## 6. Graphs and write-ups

Nothing complicated here:

- Give graphs a clear title and labelled axes.
- Add a short explanation of what each graph or result tells us.
- Include the code that produced your results.
- Save any separate figures in `reports/figures/` if needed.

All numbers in the final README and presentation should come from the final notebook, so please make sure your results can be reproduced.

## 7. Putting everything together

Once everyone has uploaded their sections, I'll combine the code, graphs, and explanations into `notebooks/olist_late_delivery.ipynb` and update the README.

We'll finalize the features and model comparisons before running the final test evaluation.

If you're unsure about anything, just ask in the group chat. The goal is for everyone to focus on their assigned work without having to worry about managing the repository.
