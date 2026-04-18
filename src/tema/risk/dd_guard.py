from typing import Union
import numpy as np
import pandas as pd


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Compute drawdown series from an equity curve (index-aware).

    Drawdown is defined as equity / running_max - 1 (so values <= 0).
    """
    if equity_curve is None or equity_curve.size == 0:
        return pd.Series(dtype=float)
    eq = pd.to_numeric(equity_curve, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if eq.empty:
        return pd.Series(dtype=float)
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    dd = dd.fillna(0.0)
    return dd


def compute_dd_guard_scalar(drawdown: pd.Series, *, max_dd: float, floor: float, recovery_halflife: int = 20) -> pd.Series:
    """Compute a drawdown-guard scalar series in [floor, 1.0].

    - When drawdown > -max_dd, target is 1.0
    - When drawdown <= -max_dd, target is `floor`
    - Scalar moves toward target via exponential smoothing with a half-life.

    Half-life semantics: after `recovery_halflife` periods the distance to the target is halved.
    """
    if drawdown is None or drawdown.size == 0:
        return pd.Series(dtype=float)
    dd = pd.to_numeric(drawdown, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # smoothing factor per period derived from half-life
    halflife = max(1, int(recovery_halflife))
    alpha = 1.0 - (0.5 ** (1.0 / float(halflife)))
    out = []
    prev = 1.0
    for v in dd.to_numpy(dtype=float):
        target = 1.0 if v > -abs(float(max_dd)) else float(floor)
        curr = prev * (1.0 - alpha) + target * alpha
        # clamp
        curr = min(max(curr, float(floor)), 1.0)
        out.append(curr)
        prev = curr
    return pd.Series(out, index=dd.index)


def apply_scalar_to_weights(weights_schedule: np.ndarray, scalar: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Apply scalar (per-period) to a weight schedule.

    weights_schedule: shape (periods, n_assets)
    scalar: length periods

    Returns adjusted schedule with same shape. Multiplication is elementwise per-row.
    """
    if weights_schedule is None:
        return np.empty((0, 0))
    w = np.asarray(weights_schedule, dtype=float)
    s = np.asarray(scalar, dtype=float)
    if w.ndim != 2:
        raise ValueError("weights_schedule must be 2D array (periods, assets)")
    if s.ndim != 1:
        s = s.reshape(-1)
    if w.shape[0] != s.shape[0]:
        raise ValueError("scalar length must match number of periods in weights_schedule")
    # multiply each row by scalar
    out = (w.T * s).T
    # ensure finite
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out
