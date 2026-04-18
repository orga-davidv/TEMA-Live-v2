import pandas as pd

from tema.signals.oos_selection import choose_best_combo_with_validation


def test_choose_best_combo_with_validation_uses_gap_penalty_score():
    subtrain = pd.Series([1.0, 2.0, 3.0], name="subtrain")
    validation = pd.Series([1.0, 1.5, 2.0], name="validation")
    combos = [(2, 5, 8), (3, 6, 9)]

    def _fake_eval(close, combo, **kwargs):
        if close.name == "subtrain":
            return {"sharpe": 2.0 if combo == (2, 5, 8) else 1.6}
        return {"sharpe": 1.0 if combo == (2, 5, 8) else 1.5}

    best_combo, info = choose_best_combo_with_validation(
        subtrain_close=subtrain,
        validation_close=validation,
        combos=combos,
        evaluate_combo=_fake_eval,
        validation_shortlist=2,
        overfit_penalty=0.5,
    )

    assert best_combo == (3, 6, 9)
    assert info["selection_score"] == 1.45
