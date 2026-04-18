from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Combo = Tuple[int, int, int]
ComboEvaluator = Callable[[pd.Series, Combo], dict[str, Any]]


def choose_best_combo_with_validation(
    subtrain_close: pd.Series,
    validation_close: pd.Series,
    combos: Sequence[Combo],
    *,
    evaluate_combo: ComboEvaluator,
    validation_shortlist: Optional[int] = 50,
    overfit_penalty: float = 0.5,
    evaluation_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[Combo, dict[str, Any]]:
    """Pick best triple-EMA combo with subtrain/validation scoring.

    Selection score:
      score = val_sharpe - overfit_penalty * abs(subtrain_sharpe - val_sharpe)

    This preserves the historical scoring used by the modular pipeline so
    fixture parity remains stable while OOS extraction is modularized.
    """
    if not combos:
        raise ValueError("combos must not be empty")
    eval_kwargs = evaluation_kwargs or {}

    train_rows: list[dict[str, float]] = []
    for combo in combos:
        r = evaluate_combo(subtrain_close, combo, **eval_kwargs)
        train_rows.append(
            {
                "ema1_period": float(combo[0]),
                "ema2_period": float(combo[1]),
                "ema3_period": float(combo[2]),
                "subtrain_sharpe": float(r["sharpe"]),
            }
        )
    train_df = pd.DataFrame(train_rows)
    rank_col = train_df["subtrain_sharpe"].replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    train_df = train_df.assign(_rank=rank_col).sort_values("_rank", ascending=False).drop(columns=["_rank"])

    shortlist_n = len(train_df) if validation_shortlist is None else max(1, min(int(validation_shortlist), len(train_df)))
    shortlist = train_df.head(shortlist_n).copy()

    val_rows: list[dict[str, float]] = []
    for _, row in shortlist.iterrows():
        combo = (int(row["ema1_period"]), int(row["ema2_period"]), int(row["ema3_period"]))
        r = evaluate_combo(validation_close, combo, **eval_kwargs)
        val_rows.append(
            {
                "ema1_period": float(combo[0]),
                "ema2_period": float(combo[1]),
                "ema3_period": float(combo[2]),
                "val_sharpe": float(r["sharpe"]),
            }
        )
    val_df = pd.DataFrame(val_rows)
    merged = shortlist.merge(val_df, on=["ema1_period", "ema2_period", "ema3_period"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping results between subtrain shortlist and validation")

    gap = (merged["subtrain_sharpe"] - merged["val_sharpe"]).abs()
    merged["selection_score"] = merged["val_sharpe"] - float(overfit_penalty) * gap
    rank_sel = merged["selection_score"].replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    rank_val = merged["val_sharpe"].replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    merged = merged.assign(_rank_sel=rank_sel, _rank_val=rank_val).sort_values(
        ["_rank_sel", "_rank_val"], ascending=[False, False]
    )

    best = merged.iloc[0]
    best_combo = (int(best["ema1_period"]), int(best["ema2_period"]), int(best["ema3_period"]))
    info = {
        "subtrain_sharpe": float(best["subtrain_sharpe"]),
        "val_sharpe": float(best["val_sharpe"]),
        "selection_score": float(best["selection_score"]),
        "shortlist_size": int(shortlist_n),
        "ranking": merged.drop(columns=["_rank_sel", "_rank_val"]).reset_index(drop=True),
    }
    return best_combo, info
