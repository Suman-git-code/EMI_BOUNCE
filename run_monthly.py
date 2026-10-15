"""
ONE-COMMAND monthly run: score new accounts, show tier summary, optionally check against real outcomes.

  python run_monthly.py new_accounts.csv
  python run_monthly.py new_accounts.csv new_accounts_actual_outcomes.csv     (also checks accuracy)

Keep this file, score_accounts.py and emi_bounce_final.pkl in the same folder.
"""
import os
import sys
import joblib
import pandas as pd
from score_accounts import score

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["HIGH", "MEDIUM", "LOW"]


def run(accounts_csv, outcomes_csv=None):
    bundle = joblib.load(os.path.join(HERE, "emi_bounce_final.pkl"))
    scored = score(pd.read_csv(accounts_csv), bundle)

    out_path = os.path.splitext(accounts_csv)[0] + "_scored.csv"
    scored.to_csv(out_path, index=False)

    print(f"\nScored {len(scored):,} accounts -> {out_path}")
    summary = scored["RISK_TIER"].value_counts().reindex(ORDER).to_frame("accounts")
    summary["share_%"] = (summary["accounts"] / len(scored) * 100).round(1)
    print(summary.to_string())
    print(f"\nHIGH-risk call list: {(scored['RISK_TIER'] == 'HIGH').sum():,} accounts (top of the output file)")

    if outcomes_csv:
        actual = pd.read_csv(outcomes_csv, dtype={"V_ACCOUNT_NUMBER": str})
        check = scored.merge(actual, on="V_ACCOUNT_NUMBER")
        t = check.groupby("RISK_TIER")["TARGET_BOUNCE"].agg(accounts="count", bounced="sum", rate="mean")
        t["bounce_rate_%"] = (t.pop("rate") * 100).round(1)
        print("\nCheck against real outcomes:")
        print(t.reindex(ORDER).to_string())
        flagged = check[check["RISK_TIER"].isin(["HIGH", "MEDIUM"])]
        caught, total = flagged["TARGET_BOUNCE"].sum(), check["TARGET_BOUNCE"].sum()
        print(f"\nHIGH + MEDIUM caught {caught:,} of {total:,} bouncers ({caught / total * 100:.1f}%)")
    return scored


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python run_monthly.py new_accounts.csv [actual_outcomes.csv]")
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
