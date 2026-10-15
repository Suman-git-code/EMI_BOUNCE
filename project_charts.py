"""
project_charts.py - draws every chart for the EMI Bounce project and saves them as PNG files.
Colours: black / grey / red / green only (no blue).

Usage in your notebook (after your models are trained):
    import sys; sys.path.append(r"C:\\EMI BOUNCE")
    import project_charts as pc
    pc.make_all(df=df, d=d, X_test=X_test, y_test=y_test, final_model=lgbm_orig,
                models={"Logistic Regression": logreg, "Random Forest": rf,
                        "Gradient Boosting": gb, "LightGBM": lgbm_orig},
                features=features, p_raw=p_raw, p_cal=p_cal, cutoff=0.609, check=check)
Or call any single function, e.g.  pc.chart_confusion(y_test, p, 0.609)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score)
from sklearn.calibration import calibration_curve

RED, GREEN, DARK, GREY = "#B03A2E", "#2E7D5B", "#222222", "#8C8C8C"
OUT = "images"
sns.set_theme(style="whitegrid")
plt.rcParams.update({"axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 100})

MONEY = ["SANCTIONED_LIMIT", "CURRENT_OUTSTANDING", "EMI_AMT",
         "AVG_BAL_PREV_MNTH", "AVG_BAL_PREV_QTR", "AVG_BAL_FY"]
OTHERS = ["LOAN_AGE_MONTHS", "UTILIZATION_PERCENTAGE", "CNT_MOV_1M", "CNT_MOV_3M", "CNT_MOV_6M"]


def _save(name, show=True):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close()
    print("saved", path)


# ---------------------------------------------------------------- EDA charts
def chart_target(df):
    c = df["TARGET_BOUNCE"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(["Paid (0)", "Bounced (1)"], c.values, color=[GREEN, RED])
    for i, v in enumerate(c.values):
        ax.text(i, v, f"{v:,}\n({v / len(df) * 100:.1f}%)", ha="center", va="bottom")
    ax.set_ylim(0, c.max() * 1.2)
    ax.set_title("Target balance: EMI bounce")
    _save("01_target_balance.png")


def chart_money_hist(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), MONEY):
        sns.histplot(df.loc[df[c] > 0, c], bins=60, log_scale=True, color=GREY, ax=ax)
        ax.set_title(c)
    fig.suptitle("Money columns (log scale): heavily right-skewed", fontweight="bold")
    _save("02_money_histograms.png")


def chart_other_hist(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), OTHERS):
        sns.histplot(df[c].dropna(), bins=50, color=GREY, ax=ax)
        ax.set_title(c)
    axes.ravel()[-1].axis("off")
    _save("03_other_histograms.png")


def chart_boxplots(df):
    cols = ["AVG_BAL_PREV_MNTH", "UTILIZATION_PERCENTAGE", "CNT_MOV_1M", "LOAN_AGE_MONTHS"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, c in zip(axes, cols):
        sns.boxplot(data=df, x="TARGET_BOUNCE", y=c, hue="TARGET_BOUNCE", legend=False,
                    palette={0: GREEN, 1: RED}, showfliers=False, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Paid", "Bounced"])
        ax.set_xlabel("")
        ax.set_title(c)
    fig.suptitle("Bouncers vs non-bouncers", fontweight="bold")
    _save("04_boxplots_by_target.png")


def chart_decile(d, col="BAL_TO_EMI", fname="05_bounce_rate_by_bal_to_emi.png"):
    bins = pd.qcut(d[col], 10, duplicates="drop")
    rate = d.groupby(bins, observed=True)["TARGET_BOUNCE"].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(rate)), rate.values, color=RED)
    for i, v in enumerate(rate.values):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.axhline(d["TARGET_BOUNCE"].mean() * 100, color=DARK, ls="--", lw=1, label="Overall bounce rate")
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels(range(1, len(rate) + 1))
    ax.set_xlabel(f"{col} (1 = lowest 10%, 10 = highest 10%)")
    ax.set_ylabel("Bounce rate %")
    ax.set_title(f"Bounce rate by {col} group")
    ax.legend()
    _save(fname)


def chart_corr(d, cols=None):
    cols = cols or (MONEY + OTHERS + ["UTILIZATION_PERCENTAGE", "LIMIT_AVAILABLE", "BAL_TO_EMI",
                                      "BAL_TREND", "MOV_MOMENTUM", "TARGET_BOUNCE"])
    cols = list(dict.fromkeys([c for c in cols if c in d.columns]))
    corr = d[cols].corr(method="spearman")
    plt.figure(figsize=(13, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdGy_r", center=0, linewidths=0.5, annot_kws={"size": 8})
    plt.title("Spearman correlation (red = rise together, grey/black = move opposite)")
    _save("06_correlation_heatmap.png")


# ------------------------------------------------------------ model charts
def chart_model_compare(models, X_test, y_test):
    rows = []
    for name, m in models.items():
        p = m.predict_proba(X_test)[:, 1]
        rows.append((name, roc_auc_score(y_test, p), average_precision_score(y_test, p)))
    r = pd.DataFrame(rows, columns=["model", "ROC-AUC", "PR-AUC"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, base in zip(axes, ["ROC-AUC", "PR-AUC"], [0.5, y_test.mean()]):
        ax.barh(r["model"], r[col], color=GREY)
        ax.axvline(base, color=RED, ls="--", lw=1, label=f"Random = {base:.2f}")
        for i, v in enumerate(r[col]):
            ax.text(v, i, f" {v:.3f}", va="center")
        ax.set_xlim(0, r[col].max() * 1.15)
        ax.set_title(col)
        ax.legend(loc="lower right")
    fig.suptitle("Model comparison on the test set (all models tied)", fontweight="bold")
    _save("07_model_comparison.png")


def chart_roc_pr(models, X_test, y_test):
    palette = [DARK, RED, GREEN, GREY, "#C98B2B"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for (name, m), colr in zip(models.items(), palette):
        p = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        pr, rc, _ = precision_recall_curve(y_test, p)
        axes[0].plot(fpr, tpr, color=colr, label=f"{name} ({roc_auc_score(y_test, p):.3f})")
        axes[1].plot(rc, pr, color=colr, label=f"{name} ({average_precision_score(y_test, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].axhline(y_test.mean(), color="k", ls="--", lw=1, label="Random")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
    for ax in axes:
        ax.legend(loc="best", fontsize=8)
    _save("08_roc_and_pr_curves.png")


def chart_confusion(y_test, p, cutoff, fname="09_confusion_matrix.png"):
    flag = (np.asarray(p) >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, flag).ravel()
    cm = np.array([[tn, fp], [fn, tp]])
    labels = [[f"TN\nPaid, not flagged\n{tn:,}", f"FP\nWasted call\n{fp:,}"],
              [f"FN\nMissed bouncer\n{fn:,}", f"TP\nCaught bouncer\n{tp:,}"]]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Greys", cbar=False, linewidths=2, linecolor="white",
                xticklabels=["Not flagged", "Flagged"], yticklabels=["Paid (0)", "Bounced (1)"],
                annot_kws={"size": 12, "fontweight": "bold"}, ax=ax)
    ax.set_xlabel("Model decision")
    ax.set_ylabel("Actual outcome")
    ax.set_title(f"Confusion matrix (score >= {cutoff:.3f}, top {flag.mean() * 100:.0f}% flagged)")
    ax.text(0.5, -0.14, f"Recall {tp / (tp + fn):.1%}  |  Precision {tp / (tp + fp):.1%}  |  "
            f"Specificity {tn / (tn + fp):.1%}", transform=ax.transAxes, ha="center", fontsize=10)
    _save(fname)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn)


def chart_threshold_tradeoff(y_test, p):
    y = np.asarray(y_test)
    order = np.argsort(-np.asarray(p))
    ys = y[order]
    pcts = np.arange(1, 101)
    n = (len(ys) * pcts / 100).astype(int)
    caught = np.cumsum(ys)[n - 1]
    recall = caught / y.sum() * 100
    precision = caught / n * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(pcts, recall, color=GREEN, lw=2, label="Recall (% of bouncers reached)")
    ax.plot(pcts, precision, color=RED, lw=2, label="Precision (% of flagged who bounce)")
    for pct in (10, 20, 30):
        ax.axvline(pct, color=GREY, ls=":", lw=1)
        ax.text(pct, 101, f"{pct}%", ha="center", fontsize=9)
    ax.set_xlabel("Share of customers flagged (riskiest first)")
    ax.set_ylabel("%")
    ax.set_title("Threshold trade-off: recall vs precision")
    ax.legend()
    _save("10_threshold_tradeoff.png")


def chart_lift_gains(y_test, p):
    y = np.asarray(y_test)
    order = np.argsort(-np.asarray(p))
    ys = y[order]
    x = np.arange(1, len(ys) + 1) / len(ys) * 100
    gains = np.cumsum(ys) / ys.sum() * 100
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(x, gains, color=RED, lw=2, label="Model")
    axes[0].plot([0, 100], [0, 100], "k--", lw=1, label="Random")
    axes[0].set(xlabel="% of customers contacted", ylabel="% of bouncers reached", title="Cumulative gains")
    axes[0].legend()
    dec = np.array_split(ys, 10)
    lift = [d_.mean() / y.mean() for d_ in dec]
    axes[1].bar(range(1, 11), lift, color=RED)
    axes[1].axhline(1, color=DARK, ls="--", lw=1)
    for i, v in enumerate(lift):
        axes[1].text(i + 1, v, f"{v:.1f}x", ha="center", va="bottom", fontsize=9)
    axes[1].set(xlabel="Risk decile (1 = riskiest)", ylabel="Lift vs random", title="Lift by decile")
    _save("11_gains_and_lift.png")


def chart_calibration(y_test, p_raw, p_cal):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for label, s, colr in [("Raw score", p_raw, RED), ("Calibrated", p_cal, GREEN)]:
        frac, mp = calibration_curve(y_test, s, n_bins=10, strategy="quantile")
        ax.plot(mp, frac, marker="o", color=colr, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.set(xlabel="Predicted probability", ylabel="Actual bounce rate", title="Calibration")
    ax.legend()
    _save("12_calibration.png")


def chart_feature_importance(model, features, top=15):
    est = model.named_steps["model"] if hasattr(model, "named_steps") else model
    imp = pd.Series(est.feature_importances_, index=features).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index, imp.values, color=DARK)
    ax.set_title("Feature importance (final model)")
    _save("13_feature_importance.png")


def chart_shap(model, X_test, n=5000):
    import shap
    sample = X_test.sample(min(n, len(X_test)), random_state=1)
    sv = shap.TreeExplainer(model)(sample)
    if sv.values.ndim == 3:
        sv = sv[:, :, 1]
    plt.figure()
    shap.plots.bar(sv, max_display=12, show=False)
    plt.title("SHAP: average impact of each feature")
    _save("14_shap_bar.png")
    plt.figure()
    shap.plots.beeswarm(sv, max_display=12, show=False, color=matplotlib.colormaps["RdGy_r"])
    plt.title("SHAP: how feature values push risk")
    _save("15_shap_beeswarm.png")
    i = int(np.argmax(model.predict_proba(sample)[:, 1]))
    plt.figure()
    shap.plots.waterfall(sv[i], show=False)
    _save("16_shap_waterfall_riskiest_account.png")


# ------------------------------------------------------- new-data check chart
def chart_tiers(check):
    """check = scored accounts merged with real outcomes (needs RISK_TIER and TARGET_BOUNCE)."""
    t = check.groupby("RISK_TIER")["TARGET_BOUNCE"].mean().reindex(["HIGH", "MEDIUM", "LOW"]) * 100
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(t.index, t.values, color=[RED, "#C98B2B", GREEN])
    for i, v in enumerate(t.values):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.axhline(check["TARGET_BOUNCE"].mean() * 100, color=DARK, ls="--", lw=1, label="Overall bounce rate")
    ax.set(ylabel="Real bounce rate %", title="Bounce rate by risk tier (new, unseen accounts)")
    ax.legend()
    _save("17_bounce_rate_by_tier.png")


# ----------------------------------------------------------------- run all
def make_all(df, d, X_test, y_test, final_model, models, features,
             p_raw=None, p_cal=None, cutoff=0.609, check=None, shap_charts=True):
    p = final_model.predict_proba(X_test)[:, 1]
    chart_target(df)
    chart_money_hist(df)
    chart_other_hist(df)
    chart_boxplots(df)
    chart_decile(d, "BAL_TO_EMI", "05_bounce_rate_by_bal_to_emi.png")
    chart_decile(d, "AVG_BAL_PREV_MNTH", "05b_bounce_rate_by_balance.png")
    chart_corr(d)
    chart_model_compare(models, X_test, y_test)
    chart_roc_pr(models, X_test, y_test)
    print(chart_confusion(y_test, p, cutoff))
    chart_threshold_tradeoff(y_test, p)
    chart_lift_gains(y_test, p)
    if p_raw is not None and p_cal is not None:
        chart_calibration(y_test, p_raw, p_cal)
    chart_feature_importance(final_model, features)
    if shap_charts:
        chart_shap(final_model, X_test)
    if check is not None:
        chart_tiers(check)
    print(f"\nAll charts saved in the folder: {os.path.abspath(OUT)}")
