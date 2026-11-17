"""Modelo global único em LightGBM para todos os subsistemas (SE, S, NE, N)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.baseline import LAG_DAYS, compute_metrics
from src.evaluate import DEFAULT_N_ESTIMATORS, DEFAULT_N_WEEKS, _safe_gain_pct, generate_weekly_folds
from src.features import GLOBAL_FEATURE_COLUMNS, build_global_features
from src.train import (
    EXPERIMENT_NAME,
    WARMUP_DAYS,
    configure_mlflow_experiment,
    make_lgbm_params,
    resolve_tracking_uri,
)

GLOBAL_GOLD_PATH = Path("data/gold/carga_energia_subsistemas.parquet")
GLOBAL_BACKTEST_PATH = Path("data/predictions/global_lgbm_backtest.parquet")
GLOBAL_FORECAST_PATH = Path("data/predictions/global_lgbm_forecast_7d.parquet")
GLOBAL_SUMMARY_PATH = Path("data/predictions/global_lgbm_summary.json")


def load_global_gold_series(path: Path = GLOBAL_GOLD_PATH) -> pd.DataFrame:
    """Carrega a série Gold de todos os subsistemas ordenada cronologicamente."""
    df = pd.read_parquet(path)
    df["data_referencia"] = pd.to_datetime(df["data_referencia"])
    return df.sort_values(["id_subsistema", "data_referencia"]).reset_index(drop=True)


def run_global_backtest(
    global_df: pd.DataFrame,
    n_weeks: int = DEFAULT_N_WEEKS,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> pd.DataFrame:
    """Executa backtesting semanal com um único LightGBM para SE, S, NE e N."""
    feat_df = build_global_features(global_df, horizon=0)
    observed = feat_df[~feat_df["is_forecast"]].copy()
    min_valid_date = observed["data_referencia"].min() + pd.Timedelta(days=WARMUP_DAYS)
    observed = observed[observed["data_referencia"] >= min_valid_date].reset_index(drop=True)

    folds = generate_weekly_folds(observed["data_referencia"].drop_duplicates(), n_weeks=n_weeks)
    feature_cols = list(GLOBAL_FEATURE_COLUMNS)
    fold_frames: list[pd.DataFrame] = []

    for fold_idx, (origin_date, test_start, test_end) in enumerate(folds, start=1):
        train_df = observed[observed["data_referencia"] <= origin_date]
        test_df = observed[
            (observed["data_referencia"] >= test_start) & (observed["data_referencia"] <= test_end)
        ].copy()
        if train_df.empty or test_df.empty:
            raise ValueError(f"Fold {fold_idx} inválido no modelo global.")
        if train_df["data_referencia"].max() >= test_df["data_referencia"].min():
            raise AssertionError("Vazamento temporal detectado no split global.")

        params = make_lgbm_params("regression_l1")
        params["n_estimators"] = n_estimators
        model = lgb.LGBMRegressor(**params)
        model.fit(train_df[feature_cols], train_df["carga_mwmed"])

        fold_frames.append(
            pd.DataFrame(
                {
                    "data_referencia": test_df["data_referencia"].to_numpy(),
                    "id_subsistema": test_df["id_subsistema"].to_numpy(),
                    "fold": fold_idx,
                    "y_true": test_df["carga_mwmed"].to_numpy(dtype=float),
                    "y_pred_baseline": test_df["lag_7"].to_numpy(dtype=float),
                    "y_pred_global_lgbm": model.predict(test_df[feature_cols]),
                }
            )
        )

    bt_df = pd.concat(fold_frames, ignore_index=True)
    bt_df["data_referencia"] = pd.to_datetime(bt_df["data_referencia"])
    return bt_df.sort_values(["id_subsistema", "data_referencia"]).reset_index(drop=True)


def build_global_summary(bt_df: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    """Consolida métricas por subsistema (SE, S, NE, N) e agregado geral."""
    by_subsystem: list[dict[str, object]] = []
    for sub_id in ("SE", "S", "NE", "N"):
        sub = bt_df[bt_df["id_subsistema"] == sub_id]
        if sub.empty:
            continue
        base_m = compute_metrics(sub["y_true"], sub["y_pred_baseline"])
        glob_m = compute_metrics(sub["y_true"], sub["y_pred_global_lgbm"])
        glob_rmse = float(np.sqrt(np.mean((sub["y_true"] - sub["y_pred_global_lgbm"]) ** 2)))
        by_subsystem.append(
            {
                "id_subsistema": sub_id,
                "n_obs": glob_m["n_obs"],
                "baseline_d7__mae": base_m["mae"],
                "baseline_d7__mape": base_m["mape"],
                "global_lgbm__mae": glob_m["mae"],
                "global_lgbm__mape": glob_m["mape"],
                "global_lgbm__rmse": glob_rmse,
                "ganho_mae_vs_d7_pct": _safe_gain_pct(base_m["mae"], glob_m["mae"]),
            }
        )

    all_base = compute_metrics(bt_df["y_true"], bt_df["y_pred_baseline"])
    all_glob = compute_metrics(bt_df["y_true"], bt_df["y_pred_global_lgbm"])
    overall = [
        {
            "n_obs": all_glob["n_obs"],
            "baseline_d7__mae": all_base["mae"],
            "baseline_d7__mape": all_base["mape"],
            "global_lgbm__mae": all_glob["mae"],
            "global_lgbm__mape": all_glob["mape"],
            "ganho_mae_vs_d7_pct": _safe_gain_pct(all_base["mae"], all_glob["mae"]),
        }
    ]
    return {"overall": overall, "by_subsystem": by_subsystem}


def forecast_global_next_7_days(
    global_df: pd.DataFrame,
    n_estimators: int = 800,
) -> pd.DataFrame:
    """Treina o modelo global até o último dia observado e prevê 7 dias por subsistema."""
    feat_df = build_global_features(global_df, horizon=LAG_DAYS)
    observed = feat_df[~feat_df["is_forecast"]].copy()
    min_valid_date = observed["data_referencia"].min() + pd.Timedelta(days=WARMUP_DAYS)
    train_df = observed[observed["data_referencia"] >= min_valid_date]
    future_df = feat_df[feat_df["is_forecast"]].copy()

    feature_cols = list(GLOBAL_FEATURE_COLUMNS)
    params = make_lgbm_params("regression_l1")
    params["n_estimators"] = n_estimators
    model = lgb.LGBMRegressor(**params)
    model.fit(train_df[feature_cols], train_df["carga_mwmed"])

    return pd.DataFrame(
        {
            "data_referencia": future_df["data_referencia"].to_numpy(),
            "id_subsistema": future_df["id_subsistema"].to_numpy(),
            "y_pred_baseline": future_df["lag_7"].to_numpy(dtype=float),
            "y_pred_global_lgbm": model.predict(future_df[feature_cols]),
        }
    )


def _write_artifacts(
    bt_df: pd.DataFrame,
    fc_df: pd.DataFrame,
    summary: dict[str, list[dict[str, object]]],
) -> None:
    GLOBAL_BACKTEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    for df, path in ((bt_df, GLOBAL_BACKTEST_PATH), (fc_df, GLOBAL_FORECAST_PATH)):
        tmp = path.with_name(path.name + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)

    tmp_json = GLOBAL_SUMMARY_PATH.with_name(GLOBAL_SUMMARY_PATH.name + ".tmp")
    tmp_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_json.replace(GLOBAL_SUMMARY_PATH)


def main() -> None:
    """Executa backtesting e previsão 7d do modelo global LightGBM."""
    parser = argparse.ArgumentParser(description="Modelo global LightGBM para SE, S, NE e N.")
    parser.add_argument("--input", type=Path, default=GLOBAL_GOLD_PATH)
    parser.add_argument("--n-weeks", type=int, default=DEFAULT_N_WEEKS)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()

    global_df = load_global_gold_series(args.input)
    bt_df = run_global_backtest(global_df, n_weeks=args.n_weeks, n_estimators=args.n_estimators)
    summary = build_global_summary(bt_df)
    fc_df = forecast_global_next_7_days(global_df)
    _write_artifacts(bt_df, fc_df, summary)

    if not args.skip_mlflow:
        tracking_uri = resolve_tracking_uri(args.tracking_uri)
        configure_mlflow_experiment(tracking_uri, EXPERIMENT_NAME)
        ov = summary["overall"][0]
        with mlflow.start_run(run_name=f"backtest_{args.n_weeks}w_global_4subs") as run:
            mlflow.log_params(
                {
                    "model_family": "LightGBM_global_4subs",
                    "objective": "regression_l1",
                    "n_weeks": args.n_weeks,
                    "n_estimators": args.n_estimators,
                }
            )
            metrics: dict[str, float] = {
                "global_mae": float(ov["global_lgbm__mae"]),
                "global_mape": float(ov["global_lgbm__mape"]),
                "global_gain_pct": float(ov["ganho_mae_vs_d7_pct"]),
            }
            for row in summary["by_subsystem"]:
                sub_id = str(row["id_subsistema"]).lower()
                metrics[f"sub_{sub_id}_mae"] = float(row["global_lgbm__mae"])
                metrics[f"sub_{sub_id}_gain_pct"] = float(row["ganho_mae_vs_d7_pct"])
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(GLOBAL_SUMMARY_PATH))
            print(
                f"[global_model] Modelo global registrado no MLflow ({tracking_uri}), "
                f"run_id={run.info.run_id}"
            )

    print("=== Backtest Modelo Global (SE, S, NE, N) vs Baseline D-7 ===")
    print(pd.DataFrame(summary["by_subsystem"]).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))


if __name__ == "__main__":
    main()
