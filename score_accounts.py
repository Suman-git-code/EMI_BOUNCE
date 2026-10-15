"""
Monthly scoring pipeline - EMI Bounce Likelihood Model
Usage:  python score_accounts.py  new_accounts.csv  emi_bounce_final.pkl  scored_accounts.csv

Steps: load model bundle -> clean -> rebuild features -> score -> calibrate -> assign tier -> save.
The cleaning and feature code MUST match training exactly.
"""
import sys
import joblib
import numpy as np
import pandas as pd

REQUIRED = [
    "V_ACCOUNT_NUMBER", "V_CUST_REF_CODE", "SANCTIONED_LIMIT", "CURRENT_OUTSTANDING",
    "EMI_AMT", "D_FIRST_DISBURSEMENT_DATE", "LOAN_AGE_MONTHS", "UTILIZATION_PERCENTAGE",
    "LIMIT_AVAILABLE", "AVG_BAL_PREV_MNTH", "AVG_BAL_PREV_QTR", "AVG_BAL_FY",
    "CNT_MOV_1M", "CNT_MOV_3M", "CNT_MOV_6M",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing columns: {missing}")

    df = df.drop_duplicates().copy()
    df["V_ACCOUNT_NUMBER"] = df["V_ACCOUNT_NUMBER"].astype(str)
    df["D_FIRST_DISBURSEMENT_DATE"] = pd.to_datetime(df["D_FIRST_DISBURSEMENT_DATE"])

    # utilization can be rebuilt exactly from two other columns
    calc_util = df["CURRENT_OUTSTANDING"] / df["SANCTIONED_LIMIT"] * 100
    df["UTILIZATION_PERCENTAGE"] = df["UTILIZATION_PERCENTAGE"].fillna(calc_util.round(2))

    # impossible FY balances (more than 10x the quarterly balance) -> missing
    ratio = df["AVG_BAL_FY"] / df["AVG_BAL_PREV_QTR"].where(df["AVG_BAL_PREV_QTR"] > 0)
    df.loc[ratio > 10, "AVG_BAL_FY"] = np.nan
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["BAL_TO_EMI"] = d["AVG_BAL_PREV_MNTH"] / d["EMI_AMT"]
    d["EMI_TO_SANCTION"] = d["EMI_AMT"] / d["SANCTIONED_LIMIT"]
    d["BAL_TREND"] = d["AVG_BAL_PREV_MNTH"] / d["AVG_BAL_PREV_QTR"].where(d["AVG_BAL_PREV_QTR"] > 0)
    d["MOV_MOMENTUM"] = (d["CNT_MOV_1M"] * 3) / d["CNT_MOV_3M"].replace(0, np.nan)
    d["EARLY_LOAN"] = (d["LOAN_AGE_MONTHS"] < 6).astype(int)
    return d.replace([np.inf, -np.inf], np.nan)


def score(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    d = add_features(clean(df))
    raw = bundle["model"].predict_proba(d[bundle["features"]])[:, 1]
    prob = bundle["calibrator"].predict(raw)

    cut_flag = bundle["cutoff_raw"]                 # top 20% on the test set
    cut_high = bundle.get("cutoff_high", cut_flag)  # top 10% on the test set
    tier = np.where(raw >= cut_high, "HIGH",
           np.where(raw >= cut_flag, "MEDIUM", "LOW"))

    out = d[["V_ACCOUNT_NUMBER", "V_CUST_REF_CODE", "EMI_AMT",
             "AVG_BAL_PREV_MNTH", "BAL_TO_EMI"]].copy()
    out["RISK_SCORE"] = raw.round(4)
    out["BOUNCE_PROBABILITY"] = prob.round(4)
    out["RISK_TIER"] = tier
    out["ACTION"] = out["RISK_TIER"].map({"HIGH": "Call before due date",
                                          "MEDIUM": "Send SMS reminder",
                                          "LOW": "No action"})
    out["SCORING_DATE"] = pd.Timestamp.today().normalize().date()
    return out.sort_values("RISK_SCORE", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    in_csv, model_pkl, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    bundle = joblib.load(model_pkl)
    scored = score(pd.read_csv(in_csv), bundle)
    scored.to_csv(out_csv, index=False)
    print(f"Scored {len(scored):,} accounts -> {out_csv}")
    print(scored["RISK_TIER"].value_counts().to_string())
