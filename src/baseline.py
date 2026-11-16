"""Baseline ingênuo D-7 (mesmo dia da semana anterior) em pandas para o subsistema SE."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

GOLD_PARQUET_PATH = Path("data/gold/carga_energia_se.parquet")
BASELINE_OUTPUT_PATH = Path("data/predictions/baseline_d7.parquet")
POST_MMGD_START = pd.Timestamp("2023-04-29")
LAG_DAYS = 7


def validate_daily_continuity(df: pd.DataFrame) -> pd.DataFrame:
    """Valida ordenação cronológica e continuidade diária sem duplicatas ou lacunas."""
    if df.empty:
        raise ValueError("Série de entrada vazia.")

    ordered = df.copy()
    ordered["data_referencia"] = pd.to_datetime(ordered["data_referencia"])
    ordered = ordered.sort_values("data_referencia").reset_index(drop=True)

    if ordered["data_referencia"].duplicated().any():
        raise ValueError("Foram encontradas datas duplicadas na série diária.")

    diffs = ordered["data_referencia"].diff().dropna()
    if not (diffs == pd.Timedelta(days=1)).all():
        raise ValueError("Foram encontradas lacunas de datas na série diária.")

    return ordered


def load_gold_series(path: Path = GOLD_PARQUET_PATH) -> pd.DataFrame:
    """Carrega a série Gold do subsistema SE e valida sua continuidade diária."""
    df = pd.read_parquet(path)
    return validate_daily_continuity(df)


def generate_baseline_history(df: pd.DataFrame) -> pd.DataFrame:
    """Gera previsões históricas D-7 alinhadas (y_pred_baseline[d] = carga[d - 7])."""
    ordered = validate_daily_continuity(df)
    if len(ordered) <= LAG_DAYS:
        raise ValueError("São necessários mais de 7 dias para gerar o histórico D-7.")

    result = pd.DataFrame(
        {
            "data_referencia": ordered["data_referencia"],
            "regime_metodologico": (
                ordered["regime_metodologico"]
                if "regime_metodologico" in ordered.columns
                else "desconhecido"
            ),
            "y_true": ordered["carga_mwmed"].astype(float),
            "y_pred_baseline": ordered["carga_mwmed"].shift(LAG_DAYS).astype(float),
            "is_forecast": False,
        }
    )
    return result.iloc[LAG_DAYS:].reset_index(drop=True)


def forecast_next_7_days(df: pd.DataFrame, horizon: int = LAG_DAYS) -> pd.DataFrame:
    """Prevê os próximos `horizon` dias (até 7) repetindo os valores de D-7."""
    if not 1 <= horizon <= LAG_DAYS:
        raise ValueError(f"O horizonte do baseline D-7 deve estar entre 1 e {LAG_DAYS} dias.")

    ordered = validate_daily_continuity(df)
    if len(ordered) < LAG_DAYS:
        raise ValueError("São necessários ao menos 7 dias observados para prever D-7.")

    last_date = ordered["data_referencia"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )
    lookup = ordered.set_index("data_referencia")["carga_mwmed"].astype(float)
    anchor_dates = future_dates - pd.Timedelta(days=LAG_DAYS)
    regime = (
        str(ordered["regime_metodologico"].iloc[-1])
        if "regime_metodologico" in ordered.columns
        else "desconhecido"
    )

    return pd.DataFrame(
        {
            "data_referencia": future_dates,
            "regime_metodologico": regime,
            "y_true": float("nan"),
            "y_pred_baseline": lookup.loc[anchor_dates].to_numpy(dtype=float),
            "is_forecast": True,
        }
    )


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calcula MAE (MWmed) e MAPE (%) entre série real e prevista."""
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Séries inválidas para cálculo de métricas.")
    if y_true.isna().any() or y_pred.isna().any():
        raise ValueError("Séries não podem conter valores nulos no cálculo de métricas.")
    if (y_true <= 0).any():
        raise ValueError("`y_true` deve ser estritamente positivo para cálculo de MAPE.")

    abs_err = (y_true - y_pred).abs()
    mae = float(abs_err.mean())
    mape = float((abs_err / y_true).mean() * 100.0)
    return {"n_obs": int(len(y_true)), "mae": mae, "mape": mape}


def summarize_periods(history: pd.DataFrame) -> list[dict[str, object]]:
    """Calcula métricas do baseline no histórico completo, pós-MMGD e últimos 365 dias."""
    max_date = history["data_referencia"].max()
    last_365_start = max_date - pd.Timedelta(days=364)

    slices = [
        ("historico_completo", history),
        ("pos_mmgd (>= 2023-04-29)", history[history["data_referencia"] >= POST_MMGD_START]),
        ("ultimos_365_dias", history[history["data_referencia"] >= last_365_start]),
    ]
    summary: list[dict[str, object]] = []
    for label, subset in slices:
        if subset.empty:
            continue
        metrics = compute_metrics(subset["y_true"], subset["y_pred_baseline"])
        summary.append(
            {
                "recorte": label,
                "inicio": subset["data_referencia"].min().date().isoformat(),
                "fim": subset["data_referencia"].max().date().isoformat(),
                **metrics,
            }
        )
    return summary


def save_baseline_output(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    output_path: Path = BASELINE_OUTPUT_PATH,
) -> Path:
    """Persiste histórico alinhado e previsão de 7 dias em Parquet de forma atômica."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([history, forecast], ignore_index=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    combined.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)
    return output_path


def main() -> None:
    """Executa o baseline D-7, imprime resumo de métricas/previsão e salva o Parquet."""
    parser = argparse.ArgumentParser(
        description="Executa o baseline ingênuo D-7 sobre a série Gold do subsistema SE."
    )
    parser.add_argument("--input", type=Path, default=GOLD_PARQUET_PATH)
    parser.add_argument("--output", type=Path, default=BASELINE_OUTPUT_PATH)
    args = parser.parse_args()

    df = load_gold_series(args.input)
    history = generate_baseline_history(df)
    forecast = forecast_next_7_days(df)
    saved_path = save_baseline_output(history, forecast, args.output)

    print("=== Métricas do Baseline D-7 (Subsistema SE) ===")
    summary_df = pd.DataFrame(summarize_periods(history))
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    print("\n=== Previsão D-7 para os próximos 7 dias ===")
    forecast_view = forecast[["data_referencia", "y_pred_baseline"]].copy()
    forecast_view["dia_semana"] = forecast_view["data_referencia"].dt.day_name()
    print(forecast_view.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"\n[baseline] Resultado salvo em: {saved_path}")


if __name__ == "__main__":
    main()
