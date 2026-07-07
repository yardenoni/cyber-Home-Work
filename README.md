# Credit-Card Fraud Detection — Critical Reproduction Study

A reproduction and **critical evaluation** of a popular credit-card fraud detection tutorial,
completed for the *Data Science in Cyber* final project. The goal is not merely to reproduce the
results, but to test whether the author's headline claims (~99.8% accuracy, ~100% recall) are
supported by sound methodology — and to rebuild the pipeline correctly.

## Project description
- **Problem:** detect fraudulent credit-card transactions in a highly imbalanced dataset (0.172% fraud).
- **What we show:** the source's *code* is largely sound, but its *headline claims* are not supported by
  evidence. We (1) faithfully reproduce the author's own pipeline and show its advertised
  "Logistic Regression: 100% recall / 94% precision" does **not** hold on the real test set, (2) expose
  the *accuracy fallacy* behind the "~99.8% accuracy" headline, and (3) re-evaluate with domain-appropriate
  metrics the source omits (PR-AUC, MCC, F2). We also credit the author's correct methodology and, as a
  counterfactual, quantify how much the common SMOTE-before-split leakage *would* have inflated results —
  an error the author actually avoided.

## Links
- **Selected source (under review):** https://github.com/anshpandey96/Credit-Card-Fraud-Detection
- **Dataset source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  (ULB Machine Learning Group — 284,807 transactions, 492 frauds)
- **Backup/secondary source referenced:** https://github.com/itsaryanchauhan/credit-card-fraud-detection

## Repository contents
| File | Description |
|------|-------------|
| `fraud_detection.ipynb` | Complete, executable analysis notebook (data loading → EDA → feature engineering → modeling → evaluation → error analysis). |
| `build_notebook.py` | Builder script that generates the notebook from reviewable Python source. |
| `report.pdf` | Full written report (8 sections + executive summary). |
| `requirements.txt` | Python dependencies. |
| `README.md` | This file. |

## Dataset setup
The CSV is not committed (it is ~150 MB). Download it and place it next to the notebook:

```bash
# Option A — Kaggle CLI (requires a Kaggle API token)
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip

# Option B — download creditcard.csv manually from the dataset page above.
```

Expected file: `creditcard.csv` in the project root.

## Execution instructions
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Then run the notebook top-to-bottom:
jupyter notebook fraud_detection.ipynb
# or execute headless:
jupyter nbconvert --to notebook --execute fraud_detection.ipynb --output fraud_detection.ipynb
```

All randomness is seeded (`RANDOM_STATE = 42`) for reproducibility.

## Key findings (summary)
- A trivial "always legit" classifier scores ~99.83% accuracy — proving accuracy is meaningless here.
- Faithfully reproducing the author's own pipeline does **not** reproduce the advertised
  "100% recall / 94% precision" for Logistic Regression; the precision claim in particular collapses.
- Judged on PR-AUC / MCC / F2, tree ensembles (Random Forest / XGBoost) outperform logistic regression,
  and threshold tuning on the precision-recall curve is essential to manage the FP/FN trade-off.
- The author's code correctly splits before resampling (no leakage) and selects by F1 — genuine strengths
  we credit; the problems are in the *claims* and *metric choices*, not the implementation.
