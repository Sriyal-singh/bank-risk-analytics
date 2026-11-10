"""Project 2 - Loan approval: WOE scorecard (logistic) vs gradient boosting + fairness audit."""
from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .utils import MODELS, OUT, ensure_dirs, make_preprocessor, savefig, standard_metrics

TARGET = "loan_default"
NUM = ["income", "credit_history_years", "debt_to_income", "late_12m", "utilization",
       "num_products", "loan_to_income", "loan_term_months", "tenure_months"]
CAT = ["employment", "city_tier"]
PROTECTED = ["gender", "age"]  # audited, never used as model inputs
APPROVAL_RATE = 0.70           # policy: approve the best-scoring 70% of applicants


# ----------------------------------------------------------------------------------------
# Weight-of-Evidence encoder + scorecard (the industry-standard, regulator-friendly model)
# ----------------------------------------------------------------------------------------
class WOEEncoder:
    def __init__(self, num_cols, cat_cols, n_bins: int = 6):
        self.num_cols, self.cat_cols, self.n_bins = list(num_cols), list(cat_cols), n_bins

    def _label_bins(self, s: pd.Series, col: str) -> pd.Series:
        if col in self.num_cols:
            e = self.edges_[col]
            codes = pd.cut(s, e, labels=False)
            names = []
            for i in range(len(e) - 1):
                lo, hi = e[i], e[i + 1]
                names.append(f"<= {hi:.4g}" if np.isneginf(lo) else
                             f"> {lo:.4g}" if np.isposinf(hi) else f"{lo:.4g} - {hi:.4g}")
            out = codes.map(lambda c: "Missing" if pd.isna(c) else names[int(c)])
            return out.astype(str)
        return s.fillna("Missing").astype(str)

    def fit(self, X: pd.DataFrame, y_bad):
        y_bad = np.asarray(y_bad)
        self.edges_, self.woe_, self.iv_, self.bins_ = {}, {}, {}, {}
        for c in self.num_cols:
            _, e = pd.qcut(X[c], q=self.n_bins, retbins=True, duplicates="drop")
            e = np.unique(e)
            e[0], e[-1] = -np.inf, np.inf
            self.edges_[c] = e
        for c in self.num_cols + self.cat_cols:
            b = self._label_bins(X[c], c)
            df = pd.DataFrame({"bin": b.values, "bad": y_bad})
            g = df.groupby("bin")["bad"].agg(bad="sum", n="count")
            g["good"] = g["n"] - g["bad"]
            bad_d = (g["bad"] + 0.5) / (g["bad"].sum() + 0.5 * len(g))
            good_d = (g["good"] + 0.5) / (g["good"].sum() + 0.5 * len(g))
            g["woe"] = np.log(good_d / bad_d)  # positive = safer than average
            g["bad_rate"] = g["bad"] / g["n"]
            self.woe_[c] = g["woe"].to_dict()
            self.iv_[c] = float(((good_d - bad_d) * g["woe"]).sum())
            self.bins_[c] = g.reset_index()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for c in self.num_cols + self.cat_cols:
            cols[c] = self._label_bins(X[c], c).map(self.woe_[c]).fillna(0.0).values
        return pd.DataFrame(cols, index=X.index)


class Scorecard:
    """Logistic regression on WOE features, scaled to points (base 600 = 50:1 good:bad odds,
    +20 points doubles the odds)."""

    def __init__(self, num_cols=NUM, cat_cols=CAT, base_score=600, base_odds=50, pdo=20, C=1.0):
        self.num_cols, self.cat_cols = list(num_cols), list(cat_cols)
        self.base_score, self.base_odds, self.pdo, self.C = base_score, base_odds, pdo, C
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    def fit(self, X, y_bad):
        self.enc_ = WOEEncoder(self.num_cols, self.cat_cols).fit(X, y_bad)
        W = self.enc_.transform(X)
        self.lr_ = LogisticRegression(C=self.C, max_iter=2000).fit(W, 1 - np.asarray(y_bad))
        return self

    def _logit_good(self, X):
        return self.lr_.decision_function(self.enc_.transform(X))

    def score(self, X) -> np.ndarray:
        return self.offset + self.factor * self._logit_good(X)

    def prob_bad(self, X) -> np.ndarray:
        return self.lr_.predict_proba(self.enc_.transform(X))[:, 0]

    def table(self) -> pd.DataFrame:
        feats = self.num_cols + self.cat_cols
        m = len(feats)
        rows = []
        for f, coef in zip(feats, self.lr_.coef_[0]):
            for _, r in self.enc_.bins_[f].iterrows():
                pts = self.factor * coef * r["woe"] + (self.offset + self.factor * self.lr_.intercept_[0]) / m
                rows.append({"feature": f, "bin": r["bin"], "customers": int(r["n"]),
                             "bad_rate": round(float(r["bad_rate"]), 4),
                             "woe": round(float(r["woe"]), 3), "points": round(float(pts), 1)})
        return pd.DataFrame(rows)

    def information_value(self) -> pd.DataFrame:
        iv = pd.Series(self.enc_.iv_, name="information_value").sort_values(ascending=False)
        return iv.round(4).reset_index().rename(columns={"index": "feature"})


# ----------------------------------------------------------------------------------------
# Fairness audit
# ----------------------------------------------------------------------------------------
def age_group(age: pd.Series) -> pd.Series:
    return pd.cut(age, [0, 29, 45, 200], labels=["<30", "30-45", "46+"]).astype(str)


def fairness_report(groups: pd.Series, approved: np.ndarray, y_bad: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"group": groups.values, "approved": approved, "bad": y_bad})
    rows = []
    for g, d in df.groupby("group"):
        good = d[d["bad"] == 0]
        appr = d[d["approved"] == 1]
        rows.append({
            "group": g, "applicants": len(d),
            "approval_rate": round(float(d["approved"].mean()), 4),
            "bad_rate_of_approved": round(float(appr["bad"].mean()), 4) if len(appr) else np.nan,
            "approval_rate_of_good_applicants": round(float(good["approved"].mean()), 4),
        })
    out = pd.DataFrame(rows)
    out["disparate_impact_vs_best"] = (out["approval_rate"] / out["approval_rate"].max()).round(3)
    return out


def run(df: pd.DataFrame, seed: int = 42, save: bool = True) -> dict:
    ensure_dirs()
    X = df[NUM + CAT].copy()
    X[NUM] = X[NUM].astype(float)
    y = df[TARGET].values
    meta = df[["gender", "age"]]
    Xtr, Xte, ytr, yte, _, mte = train_test_split(X, y, meta, test_size=0.3, stratify=y, random_state=seed)

    # 1) WOE scorecard
    sc = Scorecard().fit(Xtr, ytr)
    s_tr, s_te = sc.score(Xtr), sc.score(Xte)
    # 2) plain logistic regression and 3) gradient boosting for comparison
    lr = Pipeline([("pre", make_preprocessor(NUM, CAT)), ("m", LogisticRegression(max_iter=3000))]).fit(Xtr, ytr)
    gb = Pipeline([("pre", make_preprocessor(NUM, CAT, scale=False)),
                   ("m", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=300,
                                                         random_state=seed))]).fit(Xtr, ytr)
    p_sc, p_lr, p_gb = sc.prob_bad(Xte), lr.predict_proba(Xte)[:, 1], gb.predict_proba(Xte)[:, 1]
    metrics = {"WOE scorecard": standard_metrics(yte, p_sc),
               "Logistic regression": standard_metrics(yte, p_lr),
               "Gradient boosting": standard_metrics(yte, p_gb)}

    # approval policy: approve best 70% (cut-off set on TRAIN scores, applied to TEST)
    cutoff = float(np.quantile(s_tr, 1 - APPROVAL_RATE))
    appr_sc = (s_te >= cutoff).astype(int)
    gb_cut = float(np.quantile(gb.predict_proba(Xtr)[:, 1], APPROVAL_RATE))
    appr_gb = (p_gb <= gb_cut).astype(int)
    policy = {
        "score_cutoff": round(cutoff, 1),
        "scorecard": {"approval_rate": round(float(appr_sc.mean()), 4),
                      "bad_rate_of_approved": round(float(yte[appr_sc == 1].mean()), 4)},
        "gradient_boosting": {"approval_rate": round(float(appr_gb.mean()), 4),
                              "bad_rate_of_approved": round(float(yte[appr_gb == 1].mean()), 4)},
        "portfolio_bad_rate_if_approve_all": round(float(yte.mean()), 4),
    }

    # fairness
    audits = {}
    for label, approved in (("scorecard", appr_sc), ("gradient_boosting", appr_gb)):
        g_gender = fairness_report(mte["gender"], approved, yte)
        g_age = fairness_report(age_group(mte["age"]), approved, yte)
        g_gender.to_csv(OUT / f"fairness_gender_{label}.csv", index=False)
        g_age.to_csv(OUT / f"fairness_age_{label}.csv", index=False)
        audits[label] = {
            "gender_min_disparate_impact": float(g_gender["disparate_impact_vs_best"].min()),
            "age_min_disparate_impact": float(g_age["disparate_impact_vs_best"].min()),
            "gender_table": g_gender.to_dict("records"),
            "age_table": g_age.to_dict("records"),
        }
        flag = min(audits[label]["gender_min_disparate_impact"], audits[label]["age_min_disparate_impact"]) < 0.80
        audits[label]["four_fifths_rule_flag"] = bool(flag)

    sc.table().to_csv(OUT / "scorecard_points.csv", index=False)
    sc.information_value().to_csv(OUT / "scorecard_information_value.csv", index=False)

    # plots
    figs = []
    plt.figure(figsize=(6.2, 4.4))
    plt.hist(s_te[yte == 0], bins=40, alpha=0.6, label="Repaid", color="#27ae60")
    plt.hist(s_te[yte == 1], bins=40, alpha=0.6, label="Defaulted", color="#c0392b")
    plt.axvline(cutoff, ls="--", c="k", label=f"Cut-off {cutoff:.0f}")
    plt.xlabel("Credit score"); plt.ylabel("Applicants"); plt.title("Scorecard separates good from bad")
    plt.legend()
    figs.append(savefig("scoring_score_distribution.png"))

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for a, (title, key) in zip(ax, (("Gender", "gender_table"), ("Age group", "age_table"))):
        t = pd.DataFrame(audits["scorecard"][key])
        a.bar(t["group"], t["approval_rate"] * 100, color="#2c7fb8")
        a.axhline(APPROVAL_RATE * 100, ls="--", c="grey")
        a.set_title(f"Approval rate by {title.lower()} (scorecard)"); a.set_ylabel("%")
    figs.append(savefig("scoring_fairness_approval_rates.png"))

    if save:
        joblib.dump({"scorecard": sc, "cutoff": cutoff}, MODELS / "scorecard.joblib")

    return {
        "models": metrics,
        "approval_policy": policy,
        "fairness": {k: {kk: vv for kk, vv in v.items() if kk not in ("gender_table", "age_table")}
                     for k, v in audits.items()},
        "fairness_tables": {k: {"gender": v["gender_table"], "age": v["age_table"]} for k, v in audits.items()},
        "top_information_value": sc.information_value().head(5).to_dict("records"),
        "figures": figs,
    }
