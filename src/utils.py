"""Shared helpers: paths, metrics, preprocessing, plotting."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
MODELS = ROOT / "models"


def ensure_dirs() -> None:
    for d in (DATA, FIG, MODELS):
        d.mkdir(parents=True, exist_ok=True)


def inr(x: float) -> str:
    return f"Rs {x:,.0f}"


def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def standard_metrics(y, p) -> dict:
    auc = float(roc_auc_score(y, p))
    return {
        "roc_auc": round(auc, 4),
        "gini": round(2 * auc - 1, 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "ks": round(ks_statistic(y, p), 4),
    }


def recall_at_top(y, p, frac: float) -> float:
    y = np.asarray(y)
    k = max(1, int(len(y) * frac))
    idx = np.argsort(-np.asarray(p))[:k]
    return float(y[idx].sum() / max(y.sum(), 1))


def decile_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Rank customers by risk (decile 1 = riskiest) and show how many bad cases each captures."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False), bins, labels=False) + 1
    g = df.groupby("decile").agg(customers=("y", "size"), bad=("y", "sum"), avg_score=("p", "mean"))
    g["bad_rate"] = g["bad"] / g["customers"]
    g["cum_capture"] = g["bad"].cumsum() / g["bad"].sum()
    g["lift"] = g["bad_rate"] / df["y"].mean()
    return g.reset_index()


def make_preprocessor(num_cols, cat_cols, scale: bool = True) -> ColumnTransformer:
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), list(num_cols))]
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cat_cols))
        )
    return ColumnTransformer(transformers)


def local_reasons(pipe, rows: pd.DataFrame, ref: dict, cols, top: int = 3) -> list[str]:
    """Model-agnostic local explanation (occlusion): replace one feature at a time with a
    typical value and measure how much the predicted risk drops. Large drop = key reason."""
    out = []
    for j in range(len(rows)):
        batch = pd.concat([rows.iloc[[j]]] * (len(cols) + 1), ignore_index=True)
        for i, c in enumerate(cols):
            batch.loc[i + 1, c] = ref[c]
        p = pipe.predict_proba(batch)[:, 1]
        deltas = p[0] - p[1:]
        order = np.argsort(-deltas)[:top]
        parts = []
        for i in order:
            if deltas[i] <= 0:
                continue
            v = rows.iloc[j][cols[i]]
            v_txt = f"{v:.3g}" if isinstance(v, (int, float, np.number)) else str(v)
            parts.append(f"{cols[i]}={v_txt} (+{deltas[i] * 100:.1f}pp)")
        out.append("; ".join(parts))
    return out


def savefig(name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    path = FIG / name
    plt.savefig(path, dpi=130)
    plt.close()
    return str(path.relative_to(ROOT))
