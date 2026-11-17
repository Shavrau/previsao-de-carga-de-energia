"""Testes unitários dos modelos adicionais (AutoETS e LightGBM global)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import GLOBAL_FEATURE_COLUMNS, build_global_features
from src.global_model import build_global_summary, run_global_backtest
from src.statistical import build_ets_summary, run_ets_backtest


def _make_single_series(n_days: int = 1200, start: str = "2023-04-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D")
    idx = np.arange(n_days, dtype=float)
    carga = 45000.0 + 3000.0 * np.sin(2.0 * np.pi * idx / 7.0) + 1500.0 * np.cos(2.0 * np.pi * idx / 365.25)
    return pd.DataFrame(
        {
            "data_referencia": dates,
            "id_subsistema": "SE",
            "nom_subsistema": "Sudeste/Centro-Oeste",
            "carga_mwmed": carga,
            "regime_metodologico": "com_mmgd_pos_2023_04",
        }
    )


def _make_global_series(n_days: int = 400) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    idx = np.arange(n_days, dtype=float)
    frames: list[pd.DataFrame] = []
    for sub_id, base in (("SE", 45000.0), ("S", 12000.0), ("NE", 13000.0), ("N", 7500.0)):
        carga = base + 1200.0 * np.sin(2.0 * np.pi * idx / 7.0)
        frames.append(
            pd.DataFrame(
                {
                    "data_referencia": dates,
                    "id_subsistema": sub_id,
                    "nom_subsistema": sub_id,
                    "carga_mwmed": carga,
                    "regime_metodologico": "com_mmgd_pos_2023_04",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_auto_ets_backtest_and_summary() -> None:
    df = _make_single_series(n_days=1200)
    bt_df = run_ets_backtest(df, n_weeks=2)
    assert len(bt_df) == 14
    assert bt_df["data_referencia"].is_monotonic_increasing
    assert not bt_df["y_pred_auto_ets"].isna().any()

    summary = build_ets_summary(bt_df)
    assert len(summary["overall"]) == 2
    assert summary["overall"][0]["modelo"] == "statsforecast_auto_ets"


def test_global_features_and_backtest_all_subsystems() -> None:
    df = _make_global_series(n_days=400)
    feat_df = build_global_features(df, horizon=7)
    future = feat_df[feat_df["is_forecast"]]
    assert len(future) == 28
    assert set(future["id_subsistema"]) == {"SE", "S", "NE", "N"}
    assert not future[list(GLOBAL_FEATURE_COLUMNS)].isna().any().any()

    bt_df = run_global_backtest(df, n_weeks=2, n_estimators=15)
    assert len(bt_df) == 56
    assert set(bt_df["id_subsistema"]) == {"SE", "S", "NE", "N"}

    summary = build_global_summary(bt_df)
    assert len(summary["by_subsystem"]) == 4
