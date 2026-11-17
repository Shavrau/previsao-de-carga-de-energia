"""Segundo concorrente estatístico (AutoETS via statsforecast) no backtesting de 52 semanas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from src.baseline import GOLD_PARQUET_PATH, LAG_DAYS, POST_MMGD_START, compute_metrics, load_gold_series
from src.evaluate import DEFAULT_N_WEEKS, _safe_gain_pct, generate_weekly_folds
from src.features import build_features
from src.train import EXPERIMENT_NAME, configure_mlflow_experiment, resolve_tracking_uri

STATS_BACKTEST_PATH = Path("data/predictions/statsforecast_backtest.parquet")
STATS_FORECAST_PATH = Path("data/predictions/statsforecast_forecast_7d.parquet")
STATS_SUMMARY_PATH = Path("data/predictions/statsforecast_summary.json")


def _fit_predict_ets(train_series: pd.DataFrame, horizon: int = LAG_DAYS) -> np.ndarray:
    """Ajusta AutoETS(season_length=7) na série diária e gera `horizon` passos à frente."""
    sf_df = pd.DataFrame(
        {
            "unique_id": "SE",
            "ds": pd.to_datetime(train_series["data_referencia"]),
            "y": train_series["carga_mwmed"].astype(float),
        }
    )
    sf = StatsForecast(models=[AutoETS(season_length=LAG_DAYS)], freq="D", n_jobs=1)
    pred = sf.forecast(df=sf_df, h=horizon)
    return pred["AutoETS"].to_numpy(dtype=float)


def run_ets_backtest(
    gold_df: pd.DataFrame,
    n_weeks: int = DEFAULT_N_WEEKS,
    train_start: pd.Timestamp = POST_MMGD_START,
) -> pd.DataFrame:
    """Executa backtesting semanal de 52 semanas para o AutoETS sem vazamento."""
    feat_df = build_features(gold_df, horizon=0)
    observed = feat_df[~feat_df["is_forecast"]].copy().reset_index(drop=True)
    folds = generate_weekly_folds(observed["data_referencia"], n_weeks=n_weeks)

    fold_frames: list[pd.DataFrame] = []
    for fold_idx, (origin_date, test_start, test_end) in enumerate(folds, start=1):
        train_mask = (observed["data_referencia"] >= train_start) & (
            observed["data_referencia"] <= origin_date
        )
        test_mask = (observed["data_referencia"] >= test_start) & (
            observed["data_referencia"] <= test_end
        )
        train_df = observed[train_mask]
        test_df = observed[test_mask]
        if train_df.empty or len(test_df) != LAG_DAYS:
            raise ValueError(f"Fold {fold_idx} inválido para AutoETS.")
        if train_df["data_referencia"].max() >= test_df["data_referencia"].min():
            raise AssertionError("Vazamento temporal detectado no backtest do AutoETS.")

        y_pred_ets = _fit_predict_ets(train_df, horizon=LAG_DAYS)
        fold_frames.append(
            pd.DataFrame(
                {
                    "data_referencia": test_df["data_referencia"].to_numpy(),
                    "fold": fold_idx,
                    "horizon_day": np.arange(1, LAG_DAYS + 1, dtype=int),
                    "is_holiday": test_df["is_holiday"].to_numpy(),
                    "is_holiday_eve": test_df["is_holiday_eve"].to_numpy(),
                    "is_post_holiday": test_df["is_post_holiday"].to_numpy(),
                    "lag_7_is_holiday": test_df["lag_7_is_holiday"].to_numpy(),
                    "y_true": test_df["carga_mwmed"].to_numpy(dtype=float),
                    "y_pred_baseline": test_df["lag_7"].to_numpy(dtype=float),
                    "y_pred_auto_ets": y_pred_ets,
                }
            )
        )

    bt_df = pd.concat(fold_frames, ignore_index=True)
    bt_df["data_referencia"] = pd.to_datetime(bt_df["data_referencia"])
    return bt_df.sort_values("data_referencia").reset_index(drop=True)


def build_ets_summary(bt_df: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    """Sumariza métricas do AutoETS comparadas ao Baseline D-7."""
    base_m = compute_metrics(bt_df["y_true"], bt_df["y_pred_baseline"])
    ets_m = compute_metrics(bt_df["y_true"], bt_df["y_pred_auto_ets"])
    ets_rmse = float(np.sqrt(np.mean((bt_df["y_true"] - bt_df["y_pred_auto_ets"]) ** 2)))
    base_rmse = float(np.sqrt(np.mean((bt_df["y_true"] - bt_df["y_pred_baseline"]) ** 2)))

    overall = [
        {
            "modelo": "statsforecast_auto_ets",
            "n_obs": ets_m["n_obs"],
            "mae": ets_m["mae"],
            "mape": ets_m["mape"],
            "rmse": ets_rmse,
            "ganho_mae_vs_d7_pct": _safe_gain_pct(base_m["mae"], ets_m["mae"]),
        },
        {
            "modelo": "baseline_d7",
            "n_obs": base_m["n_obs"],
            "mae": base_m["mae"],
            "mape": base_m["mape"],
            "rmse": base_rmse,
            "ganho_mae_vs_d7_pct": 0.0,
        },
    ]

    special_mask = (
        (bt_df["is_holiday"] == 1)
        | (bt_df["is_holiday_eve"] == 1)
        | (bt_df["is_post_holiday"] == 1)
        | (bt_df["lag_7_is_holiday"] == 1)
    )
    by_special: list[dict[str, object]] = []
    for label, mask in (
        ("Feriado / Véspera / Pós / Eco D-7", special_mask),
        ("Dia Comum", ~special_mask),
    ):
        sub = bt_df[mask]
        if sub.empty:
            continue
        b = compute_metrics(sub["y_true"], sub["y_pred_baseline"])
        e = compute_metrics(sub["y_true"], sub["y_pred_auto_ets"])
        by_special.append(
            {
                "tipo_dia": label,
                "n_obs": int(len(sub)),
                "baseline_d7__mae": b["mae"],
                "auto_ets__mae": e["mae"],
                "auto_ets__mape": e["mape"],
                "auto_ets__gain_pct": _safe_gain_pct(b["mae"], e["mae"]),
            }
        )

    return {"overall": overall, "by_special_day": by_special}


def forecast_ets_next_7_days(
    gold_df: pd.DataFrame,
    train_start: pd.Timestamp = POST_MMGD_START,
) -> pd.DataFrame:
    """Treina AutoETS até a última data observada e prevê os próximos 7 dias."""
    observed = gold_df[gold_df["data_referencia"] >= train_start].copy()
    y_pred = _fit_predict_ets(observed, horizon=LAG_DAYS)
    last_date = pd.to_datetime(observed["data_referencia"]).max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=LAG_DAYS, freq="D")
    return pd.DataFrame({"data_referencia": future_dates, "y_pred_auto_ets": y_pred})


def _write_artifacts(
    bt_df: pd.DataFrame,
    fc_df: pd.DataFrame,
    summary: dict[str, list[dict[str, object]]],
) -> None:
    STATS_BACKTEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    for df, path in ((bt_df, STATS_BACKTEST_PATH), (fc_df, STATS_FORECAST_PATH)):
        tmp = path.with_name(path.name + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)

    tmp_json = STATS_SUMMARY_PATH.with_name(STATS_SUMMARY_PATH.name + ".tmp")
    tmp_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_json.replace(STATS_SUMMARY_PATH)


def main() -> None:
    """Executa avaliação temporal do AutoETS, gera previsões e registra no MLflow."""
    parser = argparse.ArgumentParser(description="Backtesting AutoETS (statsforecast) no subsistema SE.")
    parser.add_argument("--input", type=Path, default=GOLD_PARQUET_PATH)
    parser.add_argument("--n-weeks", type=int, default=DEFAULT_N_WEEKS)
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()

    gold_df = load_gold_series(args.input)
    bt_df = run_ets_backtest(gold_df, n_weeks=args.n_weeks)
    summary = build_ets_summary(bt_df)
    fc_df = forecast_ets_next_7_days(gold_df)
    _write_artifacts(bt_df, fc_df, summary)

    if not args.skip_mlflow:
        tracking_uri = resolve_tracking_uri(args.tracking_uri)
        configure_mlflow_experiment(tracking_uri, EXPERIMENT_NAME)
        ets_row = summary["overall"][0]
        with mlflow.start_run(run_name=f"backtest_{args.n_weeks}w_auto_ets") as run:
            mlflow.log_params(
                {
                    "model_family": "statsforecast_AutoETS",
                    "season_length": LAG_DAYS,
                    "window_strategy": "post_mmgd",
                    "n_weeks": args.n_weeks,
                }
            )
            mlflow.log_metrics(
                {
                    "auto_ets_mae": float(ets_row["mae"]),
                    "auto_ets_mape": float(ets_row["mape"]),
                    "auto_ets_rmse": float(ets_row["rmse"]),
                    "auto_ets_gain_pct": float(ets_row["ganho_mae_vs_d7_pct"]),
                }
            )
            mlflow.log_artifact(str(STATS_SUMMARY_PATH))
            print(f"[statistical] AutoETS registrado no MLflow ({tracking_uri}), run_id={run.info.run_id}")

    print("=== Backtest AutoETS (statsforecast) vs Baseline D-7 ===")
    print(pd.DataFrame(summary["overall"]).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))


if __name__ == "__main__":
    main()
