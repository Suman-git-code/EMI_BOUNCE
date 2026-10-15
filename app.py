"""
EMI Bounce Risk Scorer - simple web page.
Install once:  pip install streamlit
Run:           streamlit run app.py
Keep this file, score_accounts.py and emi_bounce_final.pkl in the same folder.
"""
import os
import joblib
import pandas as pd
import streamlit as st
from score_accounts import score

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["HIGH", "MEDIUM", "LOW"]

st.set_page_config(page_title="EMI Bounce Risk Scorer", layout="wide")
st.title("EMI Bounce Risk Scorer")
st.caption("Upload next month's accounts. Get a risk tier for every account. (Built on synthetic data.)")


@st.cache_resource
def load_bundle():
    return joblib.load(os.path.join(HERE, "emi_bounce_final.pkl"))


accounts_file = st.file_uploader("1. New accounts file (CSV, no TARGET_BOUNCE needed)", type="csv")
outcomes_file = st.file_uploader("2. Optional: real outcomes file, to check accuracy", type="csv")

if accounts_file is not None:
    try:
        scored = score(pd.read_csv(accounts_file), load_bundle())
    except Exception as e:
        st.error(f"Could not score the file: {e}")
        st.stop()

    counts = scored["RISK_TIER"].value_counts().reindex(ORDER).fillna(0).astype(int)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts scored", f"{len(scored):,}")
    c2.metric("HIGH (call)", f"{counts['HIGH']:,}")
    c3.metric("MEDIUM (SMS)", f"{counts['MEDIUM']:,}")
    c4.metric("LOW", f"{counts['LOW']:,}")

    st.subheader("HIGH-risk call list")
    st.dataframe(scored[scored["RISK_TIER"] == "HIGH"].head(500), use_container_width=True)

    st.download_button("Download all scored accounts (CSV)", scored.to_csv(index=False).encode(),
                       file_name="scored_accounts.csv", mime="text/csv")

    if outcomes_file is not None:
        actual = pd.read_csv(outcomes_file, dtype={"V_ACCOUNT_NUMBER": str})
        check = scored.merge(actual, on="V_ACCOUNT_NUMBER")
        t = check.groupby("RISK_TIER")["TARGET_BOUNCE"].agg(accounts="count", bounced="sum", rate="mean")
        t["bounce_rate_%"] = (t.pop("rate") * 100).round(1)
        st.subheader("Check against real outcomes")
        st.dataframe(t.reindex(ORDER), use_container_width=True)
        flagged = check[check["RISK_TIER"].isin(["HIGH", "MEDIUM"])]
        caught, total = int(flagged["TARGET_BOUNCE"].sum()), int(check["TARGET_BOUNCE"].sum())
        st.success(f"HIGH + MEDIUM caught {caught:,} of {total:,} real bouncers ({caught / total * 100:.1f}%).")
