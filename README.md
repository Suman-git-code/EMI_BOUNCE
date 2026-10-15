# EMI Bounce Likelihood Model

A machine learning project that predicts **which loan accounts are likely to miss their next EMI**, so a bank's collections team can contact the right customers *before* the due date.

**Author:** SUMAN P
**Context:** Banking internship project (Canara Bank)
**Stack:** Python, pandas, scikit-learn, LightGBM, XGBoost, SHAP, Streamlit, Oracle SQL

> **Note on data:** the bank's real data could not be shared, so this project uses a **synthetic dataset** (200,000 accounts) built to match the bank's schema. The results show that the method and pipeline work. They are **not** measurements of real customers.

---

## Problem

When an account has too little balance on the EMI due date, the EMI **bounces**. Bounces cost the bank money and push customers towards SMA and NPA status. The goal is to give every account a **risk score and probability** of bouncing, and a tier that tells the collections team what to do.

- **Type:** binary classification (`TARGET_BOUNCE` = 1 if the EMI bounced, 0 if paid)
- **Challenge:** only about 7% of accounts bounce, so the data is imbalanced and accuracy is a misleading metric

## Key results

Final model: **LightGBM** (chosen from five models that tied on this synthetic data).

| Metric (test set) | Result | Meaning |
|---|---|---|
| ROC-AUC | 0.81 | A bouncer is ranked above a non-bouncer 81% of the time |
| PR-AUC | 0.29 | About 4x better than random (0.07) |
| Top-20% capture | 61.6% | Contacting 1 customer in 5 reaches about 6 in 10 bouncers (3x lift) |

**Test on 20,000 unseen accounts:**

| Risk tier | Action | Real bounce rate |
|---|---|---|
| HIGH (top 10%) | Call before due date | 28.3% |
| MEDIUM (next 10%) | Send SMS reminder | 14.0% |
| LOW (80%) | No action | 3.3% |

The strongest signal is **`BAL_TO_EMI`** (how many EMIs the balance covers): the bounce rate falls from 25% to 0.6% across its ten groups.

## What this project covers

1. **Synthetic data generation** with realistic data-quality problems (duplicates, missing values, overdrawn accounts, outliers)
2. **Data cleaning**, including cross-checking balances to find impossible values and avoiding data leakage by filling missing values after the train/test split
3. **EDA**: class balance, skewed distributions, bouncers vs non-bouncers, correlations, bounce rate by decile
4. **Feature engineering**: `BAL_TO_EMI`, `EMI_TO_SANCTION`, `BAL_TREND`, `MOV_MOMENTUM`, `EARLY_LOAN`
5. **Model comparison** with customer-grouped, stratified cross-validation: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM
6. **Evaluation** with ROC-AUC, PR-AUC, lift and a confusion matrix, plus a threshold analysis (recall vs precision)
7. **Probability calibration** (isotonic): raw average score 35.8% brought down to the true 7.0%
8. **Explainability** with SHAP
9. **Scoring pipeline**, an Oracle table for results, and a Streamlit web page

## Project structure

```
.
├── README.md
├── generate_emi_bounce_data.py       # synthetic data generator
├── oracle_emi_bounce.sql             # source table, data-quality checks, EDA queries
├── oracle_scored_accounts.sql        # results table and collections-team queries
├── load_scored_to_oracle.py          # loads scored results into Oracle
├── score_accounts.py                 # cleaning + features + scoring + tiers
├── run_monthly.py                    # one-command monthly run
├── app.py                            # Streamlit web page
├── notebooks/
│   └── emi_bounce_analysis.ipynb     # cleaning, EDA, modelling (add your notebook here)
└── models/
    └── emi_bounce_final.pkl          # model + calibrator + feature list + cut-offs
```

## Data

| Column | Description |
|---|---|
| `V_ACCOUNT_NUMBER`, `V_CUST_REF_CODE` | Account and customer identifiers |
| `SANCTIONED_LIMIT`, `CURRENT_OUTSTANDING`, `EMI_AMT` | Loan amount, principal outstanding, monthly instalment |
| `D_FIRST_DISBURSEMENT_DATE`, `LOAN_AGE_MONTHS` | Disbursement date and tenure of the loan |
| `UTILIZATION_PERCENTAGE`, `LIMIT_AVAILABLE` | Share of the limit used, and what remains |
| `AVG_BAL_PREV_MNTH`, `AVG_BAL_PREV_QTR`, `AVG_BAL_FY` | Average balance: last month, last quarter, financial year |
| `CNT_MOV_1M`, `CNT_MOV_3M`, `CNT_MOV_6M` | Credit (CRR) movement counts over 1, 3 and 6 months |
| `TARGET_BOUNCE` | **Target:** 1 = EMI bounced, 0 = paid |

Scope: standard (regular) accounts only. Amounts are in INR.

The data is not stored in this repository because of its size. Generate it with:

```bash
python generate_emi_bounce_data.py                                   # 200,000 accounts, seed 42
python generate_emi_bounce_data.py 20000 777 new_month_full.csv 2026-10-31   # a new batch
```

## Setup

```bash
pip install pandas numpy scikit-learn lightgbm xgboost shap matplotlib seaborn joblib streamlit oracledb
```

Python 3.9 or later is recommended.

## Usage

### 1. Score a new month of accounts (one command)

Keep `run_monthly.py`, `score_accounts.py` and `emi_bounce_final.pkl` in the same folder.

```bash
python run_monthly.py new_accounts.csv
python run_monthly.py new_accounts.csv actual_outcomes.csv    # also checks accuracy
```

The input needs the 15 feature columns (no `TARGET_BOUNCE`). The output is `<input>_scored.csv` with the risk score, calibrated probability, tier and recommended action for every account.

### 2. Web page

```bash
streamlit run app.py
```

Upload an accounts file, view the High-risk call list, and download the scored file.

### 3. Oracle

1. Run `oracle_scored_accounts.sql` to create the `EMI_BOUNCE_SCORED` table.
2. Load the scored CSV with the SQL Developer **Import Data** wizard, or edit the connection lines in `load_scored_to_oracle.py` and run it.
3. Use the queries in the SQL file for the daily call list, money at stake, and month-on-month trends.

## Method notes

- **No leakage:** the train/test split is stratified and grouped by customer, and preprocessing is fitted on the training data only.
- **Imbalance:** class weights / `scale_pos_weight` during training. Models are judged by PR-AUC and lift, not accuracy.
- **Calibration:** weighting inflates raw scores, so an isotonic calibrator converts them to real probabilities. Ranking is unchanged.
- **Tiers:** HIGH = top 10% by score, MEDIUM = next 10% (to the top-20% cut-off), LOW = the rest.
- **Consistency:** `score_accounts.py` holds the one copy of the cleaning and feature code, so training and scoring cannot drift apart.

## Limitations

- The data is synthetic, so the models tie (all about 0.81 ROC-AUC). On real data, tree-based models usually show a clear advantage, and the comparison should be repeated with a time-based split.
- The exact target definition (observation window, meaning of a bounce) must be confirmed with the bank, and every feature must be known **before** the due date to avoid leakage.
- The cut-offs depend on the bank's contact capacity, cost of a bounce and cost of a reminder.
- About 39% of bouncers were still missed on the new-data test, because their balances looked healthy. No model catches everything.
- Monitor the flag rate, bounce rate by tier and feature drift each month, and retrain when performance drops.

## Author

**SUMAN P**
