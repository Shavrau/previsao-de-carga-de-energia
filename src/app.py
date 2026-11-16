"""Dashboard Streamlit enxuto para responder à pergunta do desafio (ONS SE)."""

from __future__ import annotations

import json
from pathlib import Path

import holidays
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BACKTEST_PRED_PATH = Path("data/predictions/backtest_predictions.parquet")
BACKTEST_SUMMARY_PATH = Path("data/predictions/backtest_summary.json")
FORECAST_PATH = Path("data/predictions/lgbm_forecast_7d.parquet")

WEEKDAY_PT = {
    0: "Segunda",
    1: "Terça",
    2: "Quarta",
    3: "Quinta",
    4: "Sexta",
    5: "Sábado",
    6: "Domingo",
}

st.set_page_config(
    page_title="Previsão de Carga SE — ONS",
    layout="wide",
)


def _fmt_pt(val: float, decimals: int = 1) -> str:
    return f"{val:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, dict[str, list[dict]], pd.DataFrame]:
    """Carrega predições do backtest, resumo de métricas e previsão de 7 dias."""
    bt_df = pd.read_parquet(BACKTEST_PRED_PATH)
    bt_df["data_referencia"] = pd.to_datetime(bt_df["data_referencia"])

    summary = json.loads(BACKTEST_SUMMARY_PATH.read_text(encoding="utf-8"))

    forecast_df = pd.read_parquet(FORECAST_PATH)
    forecast_df["data_referencia"] = pd.to_datetime(forecast_df["data_referencia"])
    return bt_df, summary, forecast_df


def build_series_figure(
    bt_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    best_col: str,
    show_full_backtest: bool,
) -> go.Figure:
    """Constrói gráfico de série temporal comparando Real, Baseline D-7 e LightGBM."""
    view_bt = bt_df if show_full_backtest else bt_df.tail(84)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view_bt["data_referencia"],
            y=view_bt["y_true"],
            name="Real (ONS SE)",
            line={"color": "#10b981", "width": 2.2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view_bt["data_referencia"],
            y=view_bt["y_pred_baseline"],
            name="Baseline (D-7)",
            line={"color": "#94a3b8", "width": 1.6, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view_bt["data_referencia"],
            y=view_bt[best_col],
            name="LightGBM (backtest)",
            line={"color": "#2563eb", "width": 2.0},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["data_referencia"],
            y=forecast_df["y_pred_lgbm"],
            name="Previsão LightGBM (7d)",
            mode="lines+markers",
            line={"color": "#2563eb", "width": 2.6},
            marker={"size": 7},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["data_referencia"],
            y=forecast_df["y_pred_baseline"],
            name="Previsão Baseline D-7 (7d)",
            mode="lines+markers",
            line={"color": "#94a3b8", "width": 1.6, "dash": "dot"},
            marker={"size": 5},
        )
    )
    fig.update_layout(
        height=380,
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        yaxis_title="Carga (MWmed)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hovermode="x unified",
    )
    return fig


def build_weekday_figure(dow_df: pd.DataFrame, best_name: str) -> go.Figure:
    """Constrói gráfico de barras agrupadas com MAE por dia da semana."""
    fig = go.Figure(
        data=[
            go.Bar(
                name="Baseline (D-7)",
                x=dow_df["dia_semana"],
                y=dow_df["baseline_d7__mae"],
                marker_color="#94a3b8",
            ),
            go.Bar(
                name="LightGBM",
                x=dow_df["dia_semana"],
                y=dow_df[f"{best_name}__mae"],
                marker_color="#2563eb",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        height=320,
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        yaxis_title="MAE (MWmed)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig


def main() -> None:
    """Ponto de entrada da interface Streamlit."""
    st.title("Previsão de Carga Diária — Subsistema SE (ONS)")
    st.caption("Horizonte de 7 dias | Backtesting temporal de 52 semanas (sem vazamento)")

    missing = [p for p in (BACKTEST_PRED_PATH, BACKTEST_SUMMARY_PATH, FORECAST_PATH) if not p.exists()]
    if missing:
        st.error(
            "Artefatos de avaliação não encontrados. Execute antes: "
            "`uv run python -m src.train` e `uv run python -m src.evaluate`."
        )
        return

    bt_df, summary, forecast_df = load_dashboard_data()
    overall_df = pd.DataFrame(summary["overall"])
    base_row = overall_df[overall_df["modelo"] == "baseline_d7"].iloc[0]
    best_row = overall_df[overall_df["modelo"] != "baseline_d7"].sort_values("mae").iloc[0]
    best_name = str(best_row["modelo"])
    best_col = f"y_pred_{best_name}"

    base_mae = float(base_row["mae"])
    best_mae = float(best_row["mae"])
    mae_diff = base_mae - best_mae
    gain_pct = float(best_row["ganho_mae_vs_d7_pct"])

    base_mape = float(base_row["mape"])
    best_mape = float(best_row["mape"])
    mape_diff_pp = base_mape - best_mape

    spec_df = pd.DataFrame(summary["by_special_day"])
    hol_matches = spec_df[spec_df["tipo_dia"].str.contains("Feriado", case=False, na=False)]
    holiday_row = hol_matches.iloc[0] if not hol_matches.empty else spec_df.iloc[0]
    hol_base_mae = float(holiday_row["baseline_d7__mae"])
    hol_best_mae = float(holiday_row[f"{best_name}__mae"])
    hol_gain_pct = float(holiday_row[f"{best_name}__gain_pct"])

    dt_min = bt_df["data_referencia"].min().strftime("%d/%m/%Y")
    dt_max = bt_df["data_referencia"].max().strftime("%d/%m/%Y")

    with st.container(border=True):
        st.markdown(
            "> **Pergunta do desafio:** *O seu modelo prevê melhor do que simplesmente repetir o valor "
            "do mesmo dia da semana anterior (D-7)? Se sim, quanto melhor?*\n\n"
            f"**Sim.** No backtesting temporal de **52 semanas ({int(base_row['n_obs'])} dias, {dt_min} a {dt_max})**, "
            f"o **LightGBM (`{best_name}`)** superou o D-7 com **MAE de {_fmt_pt(best_mae, 1)} vs "
            f"{_fmt_pt(base_mae, 1)} MWmed** (**-{_fmt_pt(gain_pct, 1)}%** no MAE) e **MAPE de "
            f"{_fmt_pt(best_mape, 2)}% vs {_fmt_pt(base_mape, 2)}%** (**-{_fmt_pt(mape_diff_pp, 2)} p.p.**). "
            f"Em feriados, vésperas, pós-feriados e ecos de D-7, a redução de MAE chegou a **{_fmt_pt(hol_gain_pct, 1)}%**."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "MAE — LightGBM",
        f"{_fmt_pt(best_mae, 1)} MWmed",
        delta=f"-{_fmt_pt(mae_diff, 1)} MW (-{_fmt_pt(gain_pct, 1)}%)",
        delta_color="inverse",
    )
    c2.metric(
        "MAPE — LightGBM",
        f"{_fmt_pt(best_mape, 2)}%",
        delta=f"-{_fmt_pt(mape_diff_pp, 2)} p.p. vs D-7",
        delta_color="inverse",
    )
    c3.metric(
        "MAE — Baseline D-7",
        f"{_fmt_pt(base_mae, 1)} MWmed",
        delta=f"MAPE {_fmt_pt(base_mape, 2)}%",
        delta_color="off",
        delta_arrow="off",
    )
    c4.metric(
        "Feriados e Ecos D-7 (MAE)",
        f"{_fmt_pt(hol_best_mae, 1)} MWmed",
        delta=f"-{_fmt_pt(hol_gain_pct, 1)}% vs D-7 ({_fmt_pt(hol_base_mae, 0)})",
        delta_color="inverse",
    )

    st.subheader("Comparação Temporal: Real vs LightGBM vs Baseline D-7")
    window_option = st.radio(
        "Janela de visualização",
        options=[
            "Últimas 12 semanas + Próximos 7 dias",
            "Backtest completo (52 semanas) + Próximos 7 dias",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )
    fig_series = build_series_figure(
        bt_df,
        forecast_df,
        best_col=best_col,
        show_full_backtest=window_option.startswith("Backtest completo"),
    )
    st.plotly_chart(fig_series, width="stretch", config={"displayModeBar": False})

    col_left, col_right = st.columns([1.15, 1.0])
    with col_left:
        st.subheader("Placar do Backtesting (52 semanas)")
        overall_view = pd.DataFrame(
            {
                "Modelo": overall_df["modelo"],
                "Dias": overall_df["n_obs"].astype(int),
                "MAE (MWmed)": overall_df["mae"].map(lambda v: _fmt_pt(float(v), 1)),
                "MAPE (%)": overall_df["mape"].map(lambda v: f"{_fmt_pt(float(v), 2)}%"),
                "RMSE (MWmed)": overall_df["rmse"].map(lambda v: _fmt_pt(float(v), 1)),
                "Ganho vs D-7": overall_df["ganho_mae_vs_d7_pct"].map(lambda v: f"+{_fmt_pt(float(v), 1)}%"),
            }
        )
        st.dataframe(overall_view, hide_index=True, width="stretch")

        st.subheader("Onde o ganho é maior: Feriados vs Dias Comuns")
        spec_view = pd.DataFrame(
            {
                "Tipo de Dia": spec_df["tipo_dia"],
                "Dias": spec_df["n_obs"].astype(int),
                "Baseline D-7 (MAE)": spec_df["baseline_d7__mae"].map(lambda v: _fmt_pt(float(v), 1)),
                "LightGBM (MAE)": spec_df[f"{best_name}__mae"].map(lambda v: _fmt_pt(float(v), 1)),
                "Ganho vs D-7": spec_df[f"{best_name}__gain_pct"].map(lambda v: f"+{_fmt_pt(float(v), 1)}%"),
            }
        )
        st.dataframe(spec_view, hide_index=True, width="stretch")

    with col_right:
        st.subheader("MAE por Dia da Semana")
        dow_df = pd.DataFrame(summary["by_weekday"])
        st.plotly_chart(build_weekday_figure(dow_df, best_name), width="stretch", config={"displayModeBar": False})

    st.subheader("Previsão para os Próximos 7 Dias")
    years = sorted({int(pd.Timestamp(d).year) for d in forecast_df["data_referencia"]})
    br_holidays = holidays.country_holidays("BR", years=years)
    forecast_view = pd.DataFrame(
        {
            "Data": forecast_df["data_referencia"].dt.strftime("%d/%m/%Y"),
            "Dia da Semana": forecast_df["data_referencia"].dt.day_of_week.map(WEEKDAY_PT),
            "Calendário": [
                f"Feriado: {br_holidays[d.date()]}" if d.date() in br_holidays else "Dia comum"
                for d in forecast_df["data_referencia"]
            ],
            "Baseline D-7 (MWmed)": forecast_df["y_pred_baseline"].map(lambda v: _fmt_pt(float(v), 1)),
            "LightGBM (MWmed)": forecast_df["y_pred_lgbm"].map(lambda v: _fmt_pt(float(v), 1)),
            "Diferença (MWmed)": (forecast_df["y_pred_lgbm"] - forecast_df["y_pred_baseline"]).map(
                lambda v: f"{'+' if v >= 0 else ''}{_fmt_pt(float(v), 1)}"
            ),
        }
    )
    st.dataframe(forecast_view, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
