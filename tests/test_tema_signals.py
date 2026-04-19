import pandas as pd
import pytest

from tema.signals import PythonSignalEngine, ema, generate_crossover_signal_matrix, resolve_signal_engine, tema


def test_ema_and_tema_length():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out_ema = ema(s, period=3)
    out_tema = tema(s, period=3)
    assert len(out_ema) == len(s)
    assert len(out_tema) == len(s)
    assert float(out_ema.iloc[-1]) > 0.0


def test_generate_crossover_signal_matrix_expected_shape():
    idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    df = pd.DataFrame({"asset_a": [1, 2, 3, 2, 1, 2]}, index=idx)
    sig = generate_crossover_signal_matrix(df, fast_period=2, slow_period=3, method="ema", shift_by=1)
    assert sig.shape == df.shape
    assert sig.iloc[0, 0] == 0.0
    assert set(sig["asset_a"].unique()).issubset({-1.0, 0.0, 1.0})


def test_python_signal_engine_default_shift_matches_unshifted_signals():
    idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    df = pd.DataFrame({"asset_a": [1, 2, 3, 2, 1, 2]}, index=idx)
    engine = PythonSignalEngine()
    got = engine.generate(df, fast_period=2, slow_period=3, method="ema")
    expected = generate_crossover_signal_matrix(df, fast_period=2, slow_period=3, method="ema", shift_by=0)
    pd.testing.assert_frame_equal(got, expected)


def test_resolve_signal_engine_cpp_fallback():
    engine = resolve_signal_engine(use_cpp=True, cpp_engine=None)
    assert hasattr(engine, "generate")


def test_generate_crossover_signal_matrix_rejects_invalid_periods():
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    df = pd.DataFrame({"asset_a": [1, 2, 3, 4]}, index=idx)
    with pytest.raises(ValueError, match="smaller than slow_period"):
        generate_crossover_signal_matrix(df, fast_period=5, slow_period=3)
