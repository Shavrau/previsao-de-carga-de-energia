"""Testes unitários de engenharia de atributos anti-vazamento e treino MLflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, build_features
from src.train import ExperimentConfig, refit_and_forecast_7d, run_experiment_grid


def _make_synthetic_gold(days: int = 420) -> pd.DataFrame:
    dates = pd.date_range("2025-09-01", periods=days, freq="D")
    t = np.arange(days, dtype=float)
    load = 43000.0 + 2500.0 * np.sin(2.0 * np.pi * t / 7.0) + 1500.0 * np.cos(2.0 * np.pi * t / 365.25)
    return pd.DataFrame(
        {
            "data_referencia": dates,
            "id_subsistema": "SE",
            "nom_subsistema": "Sudeste/Centro-Oeste",
            "carga_mwmed": load,
            "regime_metodologico": "com_mmgd_pos_2023_04",
        }
    )


def test_no_data_leakage_within_7_day_horizon() -> None:
    df = _make_synthetic_gold(days=420)
    cutoff_idx = 400
    target_indices = range(cutoff_idx + 1, cutoff_idx + 8)

    base_features = build_features(df, horizon=0)

    perturbed = df.copy()
    perturbed.loc[cutoff_idx + 1 :, "carga_mwmed"] = 99999.0
    perturbed_features = build_features(perturbed, horizon=0)

    pd.testing.assert_frame_equal(
        base_features.loc[target_indices, list(FEATURE_COLUMNS)],
        perturbed_features.loc[target_indices, list(FEATURE_COLUMNS)],
    )


def test_future_7_day_features_have_no_nans() -> None:
    df = _make_synthetic_gold(days=420)
    feat = build_features(df, horizon=7)
    future_rows = feat[feat["is_forecast"]]

    assert len(future_rows) == 7
    assert future_rows["carga_mwmed"].isna().all()
    assert not future_rows[list(FEATURE_COLUMNS)].isna().any().any()


def test_holiday_and_d7_holiday_echo_flags() -> None:
    dates = pd.date_range("2025-12-20", "2026-01-15", freq="D")
    df = pd.DataFrame(
        {
            "data_referencia": dates,
            "carga_mwmed": np.linspace(40000.0, 42000.0, len(dates)),
            "regime_metodologico": "com_mmgd_pos_2023_04",
        }
    )
    feat = build_features(df, horizon=0).set_index("data_referencia")

    assert feat.loc[pd.Timestamp("2026-01-01"), "is_holiday"] == 1
    assert feat.loc[pd.Timestamp("2025-12-31"), "is_holiday_eve"] == 1
    assert feat.loc[pd.Timestamp("2026-01-02"), "is_post_holiday"] == 1
    assert feat.loc[pd.Timestamp("2026-01-02"), "is_bridge_day"] == 1
    assert feat.loc[pd.Timestamp("2026-01-08"), "lag_7_is_holiday"] == 1


def test_train_and_mlflow_smoke(tmp_path: Path) -> None:
    df = _make_synthetic_gold(days=450)
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    results = run_experiment_grid(
        gold_df=df,
        tracking_uri=tracking_uri,
        experiment_name="smoke-test",
        val_days=30,
        configs=(ExperimentConfig("full_history", "regression_l1"),),
    )
    assert len(results) == 1
    assert results["val_mae"].iloc[0] >= 0

    forecast = refit_and_forecast_7d(
        gold_df=df,
        best_row=results.iloc[0],
        tracking_uri=tracking_uri,
        experiment_name="smoke-test",
        output_path=tmp_path / "forecast.parquet",
    )
    assert len(forecast) == 7
    assert (tmp_path / "forecast.parquet").exists()
