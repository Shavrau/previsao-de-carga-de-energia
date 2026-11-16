"""Testes unitários do backtesting temporal e tabelas de comparação (Etapa 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    build_comparison_tables,
    generate_weekly_folds,
    plot_backtest_results,
    run_temporal_backtest,
)
from src.train import ExperimentConfig


def _make_backtest_series(days: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    t = np.arange(days, dtype=float)
    load = 42000.0 + 2000.0 * np.sin(2.0 * np.pi * t / 7.0)
    return pd.DataFrame(
        {
            "data_referencia": dates,
            "id_subsistema": "SE",
            "nom_subsistema": "Sudeste/Centro-Oeste",
            "carga_mwmed": load,
            "regime_metodologico": "com_mmgd_pos_2023_04",
        }
    )


def test_weekly_folds_are_strictly_chronological_and_disjoint() -> None:
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    folds = generate_weekly_folds(pd.Series(dates), n_weeks=4)

    assert len(folds) == 4
    covered_days: list[pd.Timestamp] = []
    for origin_date, test_start, test_end in folds:
        assert origin_date < test_start
        assert (test_end - test_start).days == 6
        covered_days.extend(pd.date_range(test_start, test_end, freq="D"))

    assert len(covered_days) == 28
    assert len(set(covered_days)) == 28


def test_backtest_baseline_and_tables(tmp_path: Path) -> None:
    df = _make_backtest_series(days=120)
    configs = (ExperimentConfig("full_history", "regression_l1"),)
    bt_df = run_temporal_backtest(df, configs=configs, n_weeks=4, n_estimators=20)

    assert len(bt_df) == 28
    assert list(bt_df["horizon_day"]) == [1, 2, 3, 4, 5, 6, 7] * 4

    expected_lookup = df.set_index("data_referencia")["carga_mwmed"]
    for _, row in bt_df.iterrows():
        d7 = row["data_referencia"] - pd.Timedelta(days=7)
        assert row["y_pred_baseline"] == pytest.approx(expected_lookup.loc[d7])

    tables = build_comparison_tables(bt_df, configs=configs)
    assert set(tables.keys()) == {"overall", "by_horizon", "by_weekday", "by_special_day", "by_month"}
    assert len(tables["overall"]) == 2
    assert len(tables["by_horizon"]) == 7

    figs = plot_backtest_results(
        bt_df=bt_df,
        tables=tables,
        best_model_name="full_history__regression_l1",
        output_dir=tmp_path,
    )
    assert len(figs) == 2
    assert all(p.exists() for p in figs)
