"""Treinamento LightGBM e registro de experimentos no MLflow (Etapa 5)."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd

from src.baseline import (
    GOLD_PARQUET_PATH,
    POST_MMGD_START,
    compute_metrics,
    load_gold_series,
)
from src.features import FEATURE_COLUMNS, build_features

EXPERIMENT_NAME = "ons-carga-energia-se-7d"
FORECAST_OUTPUT_PATH = Path("data/predictions/lgbm_forecast_7d.parquet")
DEFAULT_DOCKER_MLFLOW_URI = "http://127.0.0.1:5000"
DEFAULT_VAL_DAYS = 180
WARMUP_DAYS = 28


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuração de janela histórica e função objetivo de uma run."""

    window_strategy: str
    objective: str

    @property
    def run_name(self) -> str:
        return f"{self.window_strategy}__{self.objective}"


DEFAULT_GRID: tuple[ExperimentConfig, ...] = (
    ExperimentConfig("full_history", "regression_l1"),
    ExperimentConfig("full_history", "regression"),
    ExperimentConfig("post_mmgd", "regression_l1"),
    ExperimentConfig("post_mmgd", "regression"),
)


def resolve_tracking_uri(preferred_uri: str | None = None) -> str:
    """Resolve URI do MLflow priorizando env/argumento, Docker local ou SQLite fallback."""
    if preferred_uri:
        return preferred_uri
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri

    try:
        with urlopen(f"{DEFAULT_DOCKER_MLFLOW_URI}/health", timeout=1.5) as resp:
            if resp.status == 200:
                return DEFAULT_DOCKER_MLFLOW_URI
    except (URLError, TimeoutError, OSError):
        pass

    mlruns_dir = Path("mlruns")
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    db_path = (mlruns_dir / "mlflow.db").resolve().as_posix()
    return f"sqlite:///{db_path}"


def configure_mlflow_experiment(tracking_uri: str, experiment_name: str = EXPERIMENT_NAME) -> str:
    """Configura tracking URI e garante experimento compatível com o modo HTTP ou SQLite."""
    mlflow.set_tracking_uri(tracking_uri)
    target_name = experiment_name
    if tracking_uri.startswith("sqlite:"):
        existing = mlflow.get_experiment_by_name(target_name)
        if existing is not None and existing.artifact_location.startswith("mlflow-artifacts:"):
            target_name = f"{experiment_name}-local"
        if mlflow.get_experiment_by_name(target_name) is None:
            local_artifacts = (Path("mlruns") / "local_artifacts").resolve()
            local_artifacts.mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment(target_name, artifact_location=local_artifacts.as_uri())
    mlflow.set_experiment(target_name)
    return target_name


def make_lgbm_params(objective: str) -> dict[str, object]:
    """Retorna hiperparâmetros consistentes do LightGBM para série diária."""
    return {
        "objective": objective,
        "n_estimators": 800,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": 1,
        "verbose": -1,
    }


def split_train_validation(
    feat_df: pd.DataFrame,
    window_strategy: str,
    val_days: int = DEFAULT_VAL_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Realiza split estritamente cronológico mantendo a mesma validação entre estratégias."""
    observed = feat_df[~feat_df["is_forecast"]].copy()
    min_valid_date = observed["data_referencia"].min() + pd.Timedelta(days=WARMUP_DAYS)
    observed = observed[observed["data_referencia"] >= min_valid_date].reset_index(drop=True)

    max_date = observed["data_referencia"].max()
    val_start = max_date - pd.Timedelta(days=val_days - 1)

    val_df = observed[observed["data_referencia"] >= val_start].copy()
    if window_strategy == "full_history":
        train_df = observed[observed["data_referencia"] < val_start].copy()
    elif window_strategy == "post_mmgd":
        train_df = observed[
            (observed["data_referencia"] >= POST_MMGD_START)
            & (observed["data_referencia"] < val_start)
        ].copy()
    else:
        raise ValueError(f"Estratégia de janela desconhecida: {window_strategy}")

    if train_df.empty or val_df.empty:
        raise ValueError("Split cronológico produziu treino ou validação vazios.")

    return train_df, val_df


def run_experiment_grid(
    gold_df: pd.DataFrame,
    tracking_uri: str,
    experiment_name: str = EXPERIMENT_NAME,
    val_days: int = DEFAULT_VAL_DAYS,
    configs: tuple[ExperimentConfig, ...] = DEFAULT_GRID,
) -> pd.DataFrame:
    """Treina a grade de modelos com early stopping cronológico e registra no MLflow."""
    configure_mlflow_experiment(tracking_uri, experiment_name)

    feat_df = build_features(gold_df, horizon=0)
    results: list[dict[str, object]] = []

    for cfg in configs:
        train_df, val_df = split_train_validation(feat_df, cfg.window_strategy, val_days=val_days)
        x_train = train_df[list(FEATURE_COLUMNS)]
        y_train = train_df["carga_mwmed"]
        x_val = val_df[list(FEATURE_COLUMNS)]
        y_val = val_df["carga_mwmed"]

        params = make_lgbm_params(cfg.objective)
        model = lgb.LGBMRegressor(**params)
        eval_metric = "l1" if cfg.objective == "regression_l1" else "l2"
        model.fit(
            x_train,
            y_train,
            eval_X=x_val,
            eval_y=y_val,
            eval_metric=eval_metric,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        val_pred = pd.Series(model.predict(x_val), index=val_df.index)
        val_metrics = compute_metrics(y_val, val_pred)
        base_metrics = compute_metrics(y_val, val_df["lag_7"])
        gain_pct = ((base_metrics["mae"] - val_metrics["mae"]) / base_metrics["mae"]) * 100.0
        best_iter = int(model.best_iteration_ or params["n_estimators"])

        with mlflow.start_run(run_name=cfg.run_name) as run:
            mlflow.log_params(
                {
                    "window_strategy": cfg.window_strategy,
                    "val_days": val_days,
                    "feature_count": len(FEATURE_COLUMNS),
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "train_start": train_df["data_referencia"].min().date().isoformat(),
                    "train_end": train_df["data_referencia"].max().date().isoformat(),
                    "val_start": val_df["data_referencia"].min().date().isoformat(),
                    "val_end": val_df["data_referencia"].max().date().isoformat(),
                    "best_iteration": best_iter,
                    **params,
                }
            )
            mlflow.log_metrics(
                {
                    "val_mae": val_metrics["mae"],
                    "val_mape": val_metrics["mape"],
                    "baseline_val_mae": base_metrics["mae"],
                    "baseline_val_mape": base_metrics["mape"],
                    "mae_gain_vs_baseline_pct": gain_pct,
                }
            )
            importance_df = pd.DataFrame(
                {
                    "feature": FEATURE_COLUMNS,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            with tempfile.TemporaryDirectory() as tmp_dir:
                imp_path = Path(tmp_dir) / "feature_importance.csv"
                importance_df.to_csv(imp_path, index=False)
                mlflow.log_artifact(str(imp_path))

            mlflow.lightgbm.log_model(
                lgb_model=model,
                name="model",
                input_example=x_val.iloc[:3],
            )
            run_id = run.info.run_id

        results.append(
            {
                "run_name": cfg.run_name,
                "run_id": run_id,
                "window_strategy": cfg.window_strategy,
                "objective": cfg.objective,
                "best_iteration": best_iter,
                "val_mae": val_metrics["mae"],
                "val_mape": val_metrics["mape"],
                "baseline_val_mae": base_metrics["mae"],
                "baseline_val_mape": base_metrics["mape"],
                "mae_gain_pct": gain_pct,
            }
        )

    return pd.DataFrame(results).sort_values("val_mae").reset_index(drop=True)


def refit_and_forecast_7d(
    gold_df: pd.DataFrame,
    best_row: pd.Series,
    tracking_uri: str,
    experiment_name: str = EXPERIMENT_NAME,
    output_path: Path = FORECAST_OUTPUT_PATH,
) -> pd.DataFrame:
    """Retreina a configuração vencedora até a última data observada e prevê 7 dias."""
    configure_mlflow_experiment(tracking_uri, experiment_name)

    feat_all = build_features(gold_df, horizon=7)
    observed = feat_all[~feat_all["is_forecast"]].copy()
    min_valid_date = observed["data_referencia"].min() + pd.Timedelta(days=WARMUP_DAYS)
    observed = observed[observed["data_referencia"] >= min_valid_date]

    window_strategy = str(best_row["window_strategy"])
    if window_strategy == "post_mmgd":
        train_df = observed[observed["data_referencia"] >= POST_MMGD_START].copy()
    else:
        train_df = observed.copy()

    objective = str(best_row["objective"])
    best_iter = max(int(best_row["best_iteration"]), 50)
    params = make_lgbm_params(objective)
    params["n_estimators"] = best_iter

    model = lgb.LGBMRegressor(**params)
    model.fit(train_df[list(FEATURE_COLUMNS)], train_df["carga_mwmed"])

    future_df = feat_all[feat_all["is_forecast"]].copy()
    future_df["y_pred_lgbm"] = model.predict(future_df[list(FEATURE_COLUMNS)])
    forecast_out = future_df[
        ["data_referencia", "y_pred_lgbm", "lag_7"]
    ].rename(columns={"lag_7": "y_pred_baseline"})

    with mlflow.start_run(run_name=f"final__{window_strategy}__{objective}"):
        mlflow.log_params(
            {
                "selected_from_run": str(best_row["run_name"]),
                "train_rows": len(train_df),
                "train_start": train_df["data_referencia"].min().date().isoformat(),
                "train_end": train_df["data_referencia"].max().date().isoformat(),
                **params,
            }
        )
        mlflow.log_metrics(
            {
                "selected_val_mae": float(best_row["val_mae"]),
                "selected_val_mape": float(best_row["val_mape"]),
            }
        )
        mlflow.lightgbm.log_model(
            lgb_model=model,
            name="model",
            input_example=train_df[list(FEATURE_COLUMNS)].iloc[:3],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    forecast_out.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)
    return forecast_out


def main() -> None:
    """Executa grade experimental no MLflow e gera previsão dos próximos 7 dias."""
    parser = argparse.ArgumentParser(
        description="Treina modelos LightGBM com lags >= 7 dias e registra no MLflow."
    )
    parser.add_argument("--input", type=Path, default=GOLD_PARQUET_PATH)
    parser.add_argument("--output", type=Path, default=FORECAST_OUTPUT_PATH)
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--val-days", type=int, default=DEFAULT_VAL_DAYS)
    args = parser.parse_args()

    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    print(f"[train] MLflow tracking URI: {tracking_uri}")

    gold_df = load_gold_series(args.input)
    results_df = run_experiment_grid(
        gold_df=gold_df,
        tracking_uri=tracking_uri,
        val_days=args.val_days,
    )

    print("\n=== Comparação na Validação Temporal (últimos 180 dias) ===")
    cols = [
        "run_name",
        "best_iteration",
        "val_mae",
        "val_mape",
        "baseline_val_mae",
        "baseline_val_mape",
        "mae_gain_pct",
    ]
    print(results_df[cols].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    best_row = results_df.iloc[0]
    forecast_df = refit_and_forecast_7d(
        gold_df=gold_df,
        best_row=best_row,
        tracking_uri=tracking_uri,
        output_path=args.output,
    )

    print(f"\n=== Previsão LightGBM ({best_row['run_name']}) — Próximos 7 dias ===")
    print(forecast_df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"\n[train] Previsão salva em: {args.output}")


if __name__ == "__main__":
    main()
