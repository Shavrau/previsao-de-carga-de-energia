"""Testes unitários do baseline D-7."""

from __future__ import annotations

import pandas as pd
import pytest

from src.baseline import (
    compute_metrics,
    forecast_next_7_days,
    generate_baseline_history,
    validate_daily_continuity,
)


def _make_sample_series(days: int = 21) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "data_referencia": dates,
            "id_subsistema": "SE",
            "nom_subsistema": "Sudeste/Centro-Oeste",
            "carga_mwmed": [40000.0 + i * 100.0 for i in range(days)],
            "regime_metodologico": "com_mmgd_pos_2023_04",
        }
    )


def test_baseline_d7_matches_previous_week() -> None:
    df = _make_sample_series(days=21)
    history = generate_baseline_history(df)

    assert len(history) == 14
    assert history["data_referencia"].iloc[0] == pd.Timestamp("2026-01-08")

    expected_by_date = df.set_index("data_referencia")["carga_mwmed"]
    for _, row in history.iterrows():
        prev_week_date = row["data_referencia"] - pd.Timedelta(days=7)
        assert row["y_pred_baseline"] == pytest.approx(expected_by_date.loc[prev_week_date])
        assert row["is_forecast"] is False or bool(row["is_forecast"]) is False


def test_forecast_next_7_days() -> None:
    df = _make_sample_series(days=21)
    forecast = forecast_next_7_days(df, horizon=7)

    assert len(forecast) == 7
    assert list(forecast["data_referencia"]) == list(
        pd.date_range("2026-01-22", periods=7, freq="D")
    )
    assert list(forecast["y_pred_baseline"]) == list(df["carga_mwmed"].iloc[-7:])
    assert forecast["y_true"].isna().all()
    assert forecast["is_forecast"].all()


def test_validate_daily_continuity_rejects_date_gaps() -> None:
    df = _make_sample_series(days=15).drop(index=5).reset_index(drop=True)
    with pytest.raises(ValueError, match="lacunas"):
        validate_daily_continuity(df)


def test_compute_metrics() -> None:
    y_true = pd.Series([100.0, 200.0])
    y_pred = pd.Series([110.0, 180.0])

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["n_obs"] == 2
    assert metrics["mae"] == pytest.approx(15.0)
    assert metrics["mape"] == pytest.approx(10.0)
