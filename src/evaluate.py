"""Backtesting temporal (expanding window semanal), comparação com baseline e gráficos (Etapa 6)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
import mlflow
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from src.baseline import GOLD_PARQUET_PATH, POST_MMGD_START, compute_metrics, load_gold_series
from src.features import FEATURE_COLUMNS, build_features
from src.train import (
    EXPERIMENT_NAME,
    FORECAST_OUTPUT_PATH,
    WARMUP_DAYS,
    ExperimentConfig,
    configure_mlflow_experiment,
    make_lgbm_params,
    resolve_tracking_uri,
)

BACKTEST_CONFIGS: tuple[ExperimentConfig, ...] = (
    ExperimentConfig("post_mmgd", "regression"),
    ExperimentConfig("full_history", "regression_l1"),
)
BACKTEST_PREDICTIONS_PATH = Path("data/predictions/backtest_predictions.parquet")
BACKTEST_SUMMARY_PATH = Path("data/predictions/backtest_summary.json")
FIGURES_DIR = Path("docs/figures")
DEFAULT_N_WEEKS = 52
DEFAULT_N_ESTIMATORS = 150

WEEKDAY_LABELS = {
    0: "Seg",
    1: "Ter",
    2: "Qua",
    3: "Qui",
    4: "Sex",
    5: "Sáb",
    6: "Dom",
}


def generate_weekly_folds(
    dates: pd.Series,
    n_weeks: int = DEFAULT_N_WEEKS,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Gera `n_weeks` blocos cronológicos consecutivos de 7 dias ao final da série."""
    if n_weeks < 1:
        raise ValueError("`n_weeks` deve ser >= 1.")

    ordered_dates = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    max_date = ordered_dates.iloc[-1]
    min_date = ordered_dates.iloc[0]

    total_test_days = n_weeks * 7
    first_test_start = max_date - pd.Timedelta(days=total_test_days - 1)
    if first_test_start <= min_date + pd.Timedelta(days=WARMUP_DAYS):
        raise ValueError("Histórico insuficiente para o número de semanas solicitado no backtest.")

    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for k in range(n_weeks - 1, -1, -1):
        test_end = max_date - pd.Timedelta(days=7 * k)
        test_start = test_end - pd.Timedelta(days=6)
        origin_date = test_start - pd.Timedelta(days=1)
        folds.append((origin_date, test_start, test_end))
    return folds


def run_temporal_backtest(
    gold_df: pd.DataFrame,
    configs: tuple[ExperimentConfig, ...] = BACKTEST_CONFIGS,
    n_weeks: int = DEFAULT_N_WEEKS,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> pd.DataFrame:
    """Executa backtesting temporal com expanding window semanal sem vazamento."""
    feat_df = build_features(gold_df, horizon=0)
    observed = feat_df[~feat_df["is_forecast"]].copy()
    min_valid_date = observed["data_referencia"].min() + pd.Timedelta(days=WARMUP_DAYS)
    observed = observed[observed["data_referencia"] >= min_valid_date].reset_index(drop=True)

    folds = generate_weekly_folds(observed["data_referencia"], n_weeks=n_weeks)
    fold_frames: list[pd.DataFrame] = []

    for fold_idx, (origin_date, test_start, test_end) in enumerate(folds, start=1):
        test_mask = (observed["data_referencia"] >= test_start) & (
            observed["data_referencia"] <= test_end
        )
        test_df = observed[test_mask].copy()
        if len(test_df) != 7:
            raise ValueError(f"Fold {fold_idx} possui {len(test_df)} dias em vez de 7.")

        fold_out = pd.DataFrame(
            {
                "data_referencia": test_df["data_referencia"].to_numpy(),
                "fold": fold_idx,
                "origin_date": origin_date,
                "horizon_day": np.arange(1, 8, dtype=int),
                "day_of_week": test_df["day_of_week"].to_numpy(),
                "is_weekend": test_df["is_weekend"].to_numpy(),
                "is_holiday": test_df["is_holiday"].to_numpy(),
                "is_holiday_eve": test_df["is_holiday_eve"].to_numpy(),
                "is_post_holiday": test_df["is_post_holiday"].to_numpy(),
                "lag_7_is_holiday": test_df["lag_7_is_holiday"].to_numpy(),
                "y_true": test_df["carga_mwmed"].to_numpy(dtype=float),
                "y_pred_baseline": test_df["lag_7"].to_numpy(dtype=float),
            }
        )

        for cfg in configs:
            if cfg.window_strategy == "full_history":
                train_mask = observed["data_referencia"] <= origin_date
            elif cfg.window_strategy == "post_mmgd":
                train_mask = (observed["data_referencia"] >= POST_MMGD_START) & (
                    observed["data_referencia"] <= origin_date
                )
            else:
                raise ValueError(f"Estratégia desconhecida: {cfg.window_strategy}")

            train_df = observed[train_mask]
            if train_df.empty:
                raise ValueError(f"Treino vazio no fold {fold_idx} ({cfg.run_name}).")
            if train_df["data_referencia"].max() >= test_df["data_referencia"].min():
                raise AssertionError("Vazamento temporal detectado no split do backtest.")

            params = make_lgbm_params(cfg.objective)
            params["n_estimators"] = n_estimators
            model = lgb.LGBMRegressor(**params)
            model.fit(train_df[list(FEATURE_COLUMNS)], train_df["carga_mwmed"])
            fold_out[f"y_pred_{cfg.run_name}"] = model.predict(test_df[list(FEATURE_COLUMNS)])

        fold_frames.append(fold_out)

    bt_df = pd.concat(fold_frames, ignore_index=True)
    bt_df["data_referencia"] = pd.to_datetime(bt_df["data_referencia"])
    return bt_df.sort_values("data_referencia").reset_index(drop=True)


def _safe_gain_pct(base_mae: float, model_mae: float) -> float:
    """Calcula ganho percentual sobre o baseline evitando divisão por zero."""
    if base_mae == 0.0:
        return 0.0
    return ((base_mae - model_mae) / base_mae) * 100.0


def _evaluate_slice(df: pd.DataFrame, model_cols: dict[str, str]) -> dict[str, float]:
    """Calcula MAE e MAPE de todos os modelos em um subconjunto de linhas."""
    row: dict[str, float] = {"n_obs": int(len(df))}
    base_mae: float | None = None
    for label, col in model_cols.items():
        metrics = compute_metrics(df["y_true"], df[col])
        row[f"{label}__mae"] = metrics["mae"]
        row[f"{label}__mape"] = metrics["mape"]
        if col == "y_pred_baseline":
            base_mae = metrics["mae"]
    if base_mae is not None:
        for label, col in model_cols.items():
            if col != "y_pred_baseline":
                row[f"{label}__gain_pct"] = _safe_gain_pct(base_mae, row[f"{label}__mae"])
    return row


def build_comparison_tables(
    bt_df: pd.DataFrame,
    configs: tuple[ExperimentConfig, ...] = BACKTEST_CONFIGS,
) -> dict[str, pd.DataFrame]:
    """Constrói tabelas comparativas global, por horizonte, dia da semana, feriado e mês."""
    model_cols: dict[str, str] = {"baseline_d7": "y_pred_baseline"}
    for cfg in configs:
        model_cols[cfg.run_name] = f"y_pred_{cfg.run_name}"

    overall_rows: list[dict[str, object]] = []
    base_metrics = compute_metrics(bt_df["y_true"], bt_df["y_pred_baseline"])
    for label, col in model_cols.items():
        m = compute_metrics(bt_df["y_true"], bt_df[col])
        rmse = float(np.sqrt(np.mean((bt_df["y_true"] - bt_df[col]) ** 2)))
        gain_pct = _safe_gain_pct(base_metrics["mae"], m["mae"])
        overall_rows.append(
            {
                "modelo": label,
                "n_obs": m["n_obs"],
                "mae": m["mae"],
                "mape": m["mape"],
                "rmse": rmse,
                "ganho_mae_vs_d7_pct": gain_pct,
            }
        )
    overall_df = pd.DataFrame(overall_rows).sort_values("mae").reset_index(drop=True)

    by_horizon_rows = [
        {"horizon_day": int(h), **_evaluate_slice(grp, model_cols)}
        for h, grp in bt_df.groupby("horizon_day", sort=True)
    ]

    by_weekday_rows = [
        {
            "day_of_week": int(dow),
            "dia_semana": WEEKDAY_LABELS[int(dow)],
            **_evaluate_slice(grp, model_cols),
        }
        for dow, grp in bt_df.groupby("day_of_week", sort=True)
    ]

    special_mask = (
        (bt_df["is_holiday"] == 1)
        | (bt_df["is_holiday_eve"] == 1)
        | (bt_df["is_post_holiday"] == 1)
        | (bt_df["lag_7_is_holiday"] == 1)
    )
    special_series = pd.Series(
        np.where(special_mask, "Feriado / Véspera / Pós / Eco D-7", "Dia Comum"),
        index=bt_df.index,
    )
    by_special_rows = [
        {"tipo_dia": str(kind), **_evaluate_slice(grp, model_cols)}
        for kind, grp in bt_df.groupby(special_series, sort=False)
    ]

    month_series = bt_df["data_referencia"].dt.strftime("%Y-%m")
    by_month_rows = [
        {"mes": str(month_key), **_evaluate_slice(grp, model_cols)}
        for month_key, grp in bt_df.groupby(month_series, sort=True)
    ]

    return {
        "overall": overall_df,
        "by_horizon": pd.DataFrame(by_horizon_rows),
        "by_weekday": pd.DataFrame(by_weekday_rows),
        "by_special_day": pd.DataFrame(by_special_rows),
        "by_month": pd.DataFrame(by_month_rows),
    }


def plot_backtest_results(
    bt_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    best_model_name: str,
    forecast_df: pd.DataFrame | None = None,
    output_dir: Path = FIGURES_DIR,
) -> list[Path]:
    """Gera gráficos PNG comparando série real, LightGBM, Baseline D-7 e erros por perfil."""
    output_dir.mkdir(parents=True, exist_ok=True)
    best_col = f"y_pred_{best_model_name}"
    saved_paths: list[Path] = []

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    # Gráfico 1: últimos 90 dias do backtest + previsão dos próximos 7 dias
    recent = bt_df.tail(90).copy()
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=150)
    ax.plot(
        recent["data_referencia"],
        recent["y_true"],
        label="Carga Real (ONS SE)",
        color="#1e293b",
        linewidth=2.0,
    )
    ax.plot(
        recent["data_referencia"],
        recent["y_pred_baseline"],
        label="Baseline (D-7)",
        color="#94a3b8",
        linestyle="--",
        linewidth=1.4,
    )
    ax.plot(
        recent["data_referencia"],
        recent[best_col],
        label=f"LightGBM ({best_model_name})",
        color="#0284c7",
        linewidth=1.8,
    )

    if forecast_df is not None and not forecast_df.empty:
        cutoff = recent["data_referencia"].max()
        ax.axvline(cutoff, color="#64748b", linestyle=":", linewidth=1.2)
        ax.axvspan(
            cutoff,
            pd.to_datetime(forecast_df["data_referencia"]).max(),
            color="#e0f2fe",
            alpha=0.45,
            label="Previsão próximos 7 dias",
        )
        ax.plot(
            pd.to_datetime(forecast_df["data_referencia"]),
            forecast_df["y_pred_baseline"],
            color="#94a3b8",
            linestyle="--",
            marker="o",
            markersize=4,
            linewidth=1.3,
        )
        ax.plot(
            pd.to_datetime(forecast_df["data_referencia"]),
            forecast_df["y_pred_lgbm"],
            color="#0284c7",
            linestyle="-",
            marker="o",
            markersize=4.5,
            linewidth=1.9,
        )

    ax.set_title("Subsistema SE: Carga Real vs Baseline (D-7) vs LightGBM (90d + 7d futuros)")
    ax.set_ylabel("Carga Diária (MWmed)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", ncol=2, frameon=True)
    fig.tight_layout()

    series_path = output_dir / "backtest_series_and_forecast.png"
    fig.savefig(series_path, dpi=150)
    plt.close(fig)
    saved_paths.append(series_path)

    # Gráfico 2: Comparação de MAE por dia da semana e por tipo de dia (Feriados vs Comuns)
    fig, (ax_dow, ax_hol) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    dow_df = tables["by_weekday"]
    x_dow = np.arange(len(dow_df))
    width = 0.36
    ax_dow.bar(
        x_dow - width / 2,
        dow_df["baseline_d7__mae"],
        width,
        label="Baseline (D-7)",
        color="#94a3b8",
    )
    ax_dow.bar(
        x_dow + width / 2,
        dow_df[f"{best_model_name}__mae"],
        width,
        label=f"LightGBM ({best_model_name})",
        color="#0284c7",
    )
    ax_dow.set_xticks(x_dow)
    ax_dow.set_xticklabels(dow_df["dia_semana"])
    max_dow = float(
        max(dow_df["baseline_d7__mae"].max(), dow_df[f"{best_model_name}__mae"].max())
    )
    ax_dow.set_ylim(0, max_dow * 1.2)
    ax_dow.set_title("MAE por Dia da Semana (Backtest 52 semanas)")
    ax_dow.set_ylabel("MAE (MWmed)")
    ax_dow.grid(axis="y", linestyle="--", alpha=0.35)
    ax_dow.legend(loc="upper right", frameon=True)

    hol_df = tables["by_special_day"]
    x_hol = np.arange(len(hol_df))
    bars_base = ax_hol.bar(
        x_hol - width / 2,
        hol_df["baseline_d7__mae"],
        width,
        label="Baseline (D-7)",
        color="#94a3b8",
    )
    bars_lgb = ax_hol.bar(
        x_hol + width / 2,
        hol_df[f"{best_model_name}__mae"],
        width,
        label=f"LightGBM ({best_model_name})",
        color="#0284c7",
    )
    ax_hol.set_xticks(x_hol)
    ax_hol.set_xticklabels(["Dia Comum" if "Comum" in t else "Feriado / Eco D-7" for t in hol_df["tipo_dia"]])
    ax_hol.set_title("MAE: Dias Comuns vs Feriados e Ecos")
    ax_hol.set_ylabel("MAE (MWmed)")
    ax_hol.grid(axis="y", linestyle="--", alpha=0.35)
    for bars in (bars_base, bars_lgb):
        for bar in bars:
            height = bar.get_height()
            ax_hol.annotate(
                f"{height:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    fig.tight_layout()
    breakdown_path = output_dir / "error_comparison_breakdown.png"
    fig.savefig(breakdown_path, dpi=150)
    plt.close(fig)
    saved_paths.append(breakdown_path)

    return saved_paths


def save_backtest_artifacts(
    bt_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    predictions_path: Path = BACKTEST_PREDICTIONS_PATH,
    summary_path: Path = BACKTEST_SUMMARY_PATH,
) -> None:
    """Salva previsões do backtesting em Parquet e resumo JSON de forma atômica."""
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_pq = predictions_path.with_name(predictions_path.name + ".tmp")
    bt_df.to_parquet(tmp_pq, index=False)
    tmp_pq.replace(predictions_path)

    payload = {name: df.to_dict(orient="records") for name, df in tables.items()}
    tmp_json = summary_path.with_name(summary_path.name + ".tmp")
    tmp_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_json.replace(summary_path)


def log_backtest_to_mlflow(
    tables: dict[str, pd.DataFrame],
    figure_paths: list[Path],
    summary_path: Path,
    tracking_uri: str,
    n_weeks: int,
    n_estimators: int,
) -> str:
    """Registra métricas de backtesting e gráficos no MLflow."""
    configure_mlflow_experiment(tracking_uri, EXPERIMENT_NAME)

    overall = tables["overall"]
    base_row = overall[overall["modelo"] == "baseline_d7"].iloc[0]

    with mlflow.start_run(run_name=f"backtest_{n_weeks}w") as run:
        mlflow.log_params(
            {
                "evaluation_type": "expanding_window_weekly_backtest",
                "n_weeks": n_weeks,
                "n_days": int(base_row["n_obs"]),
                "n_estimators": n_estimators,
            }
        )
        metrics: dict[str, float] = {
            "baseline_mae": float(base_row["mae"]),
            "baseline_mape": float(base_row["mape"]),
        }
        for _, row in overall.iterrows():
            name = str(row["modelo"])
            if name == "baseline_d7":
                continue
            metrics[f"{name}_mae"] = float(row["mae"])
            metrics[f"{name}_mape"] = float(row["mape"])
            metrics[f"{name}_gain_pct"] = float(row["ganho_mae_vs_d7_pct"])
        mlflow.log_metrics(metrics)

        if summary_path.exists():
            mlflow.log_artifact(str(summary_path))
        for fig_path in figure_paths:
            mlflow.log_artifact(str(fig_path))
        return run.info.run_id


def main() -> None:
    """Executa backtesting completo de 52 semanas, gera gráficos e registra no MLflow."""
    parser = argparse.ArgumentParser(
        description="Avaliação por backtesting temporal (LightGBM vs Baseline D-7)."
    )
    parser.add_argument("--input", type=Path, default=GOLD_PARQUET_PATH)
    parser.add_argument("--forecast", type=Path, default=FORECAST_OUTPUT_PATH)
    parser.add_argument("--n-weeks", type=int, default=DEFAULT_N_WEEKS)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()

    gold_df = load_gold_series(args.input)
    bt_df = run_temporal_backtest(
        gold_df=gold_df,
        configs=BACKTEST_CONFIGS,
        n_weeks=args.n_weeks,
        n_estimators=args.n_estimators,
    )
    tables = build_comparison_tables(bt_df, configs=BACKTEST_CONFIGS)
    save_backtest_artifacts(bt_df, tables)

    best_lgbm = tables["overall"][tables["overall"]["modelo"] != "baseline_d7"].iloc[0]
    best_name = str(best_lgbm["modelo"])
    forecast_df = pd.read_parquet(args.forecast) if args.forecast.exists() else None
    figure_paths = plot_backtest_results(
        bt_df=bt_df,
        tables=tables,
        best_model_name=best_name,
        forecast_df=forecast_df,
    )

    if not args.skip_mlflow:
        tracking_uri = resolve_tracking_uri(args.tracking_uri)
        run_id = log_backtest_to_mlflow(
            tables=tables,
            figure_paths=figure_paths,
            summary_path=BACKTEST_SUMMARY_PATH,
            tracking_uri=tracking_uri,
            n_weeks=args.n_weeks,
            n_estimators=args.n_estimators,
        )
        print(f"[evaluate] Backtest registrado no MLflow ({tracking_uri}), run_id={run_id}")

    print(
        f"\n=== Backtest Temporal ({args.n_weeks} semanas: "
        f"{bt_df['data_referencia'].min().date()} a {bt_df['data_referencia'].max().date()}) ==="
    )
    print(tables["overall"].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("\n=== MAE por Tipo de Dia ===")
    print(tables["by_special_day"].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("\n=== MAE por Horizonte (h = 1..7 dias) ===")
    print(tables["by_horizon"].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(f"\n[evaluate] Gráficos salvos em: {[str(p) for p in figure_paths]}")


if __name__ == "__main__":
    main()
