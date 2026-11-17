"""Dashboard Streamlit enxuto e focado na resposta ao desafio (ONS SE)."""

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
STATS_BACKTEST_PATH = Path("data/predictions/statsforecast_backtest.parquet")
STATS_SUMMARY_PATH = Path("data/predictions/statsforecast_summary.json")
STATS_FORECAST_PATH = Path("data/predictions/statsforecast_forecast_7d.parquet")
GLOBAL_SUMMARY_PATH = Path("data/predictions/global_lgbm_summary.json")

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


def _inject_typography_polish() -> None:
    """Ajustes leves de espaçamento e alinhamento numérico tabular."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1320px;
        }
        [data-testid="stMetricValue"] {
            font-family: 'IBM Plex Mono', monospace;
            font-variant-numeric: tabular-nums;
            letter-spacing: -0.02em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, dict[str, list[dict]], pd.DataFrame]:
    """Carrega predições do backtest, resumo de métricas e previsão de 7 dias."""
    bt_df = pd.read_parquet(BACKTEST_PRED_PATH)
    bt_df["data_referencia"] = pd.to_datetime(bt_df["data_referencia"])

    summary = json.loads(BACKTEST_SUMMARY_PATH.read_text(encoding="utf-8"))

    forecast_df = pd.read_parquet(FORECAST_PATH)
    forecast_df["data_referencia"] = pd.to_datetime(forecast_df["data_referencia"])
    return bt_df, summary, forecast_df


@st.cache_data
def load_bonus_data() -> tuple[
    pd.DataFrame | None,
    dict | None,
    pd.DataFrame | None,
    dict | None,
]:
    """Carrega resultados opcionais de AutoETS (statsforecast) e modelo global."""
    stats_bt = pd.read_parquet(STATS_BACKTEST_PATH) if STATS_BACKTEST_PATH.exists() else None
    if stats_bt is not None:
        stats_bt["data_referencia"] = pd.to_datetime(stats_bt["data_referencia"])

    stats_summary = (
        json.loads(STATS_SUMMARY_PATH.read_text(encoding="utf-8"))
        if STATS_SUMMARY_PATH.exists()
        else None
    )
    stats_fc = pd.read_parquet(STATS_FORECAST_PATH) if STATS_FORECAST_PATH.exists() else None
    if stats_fc is not None:
        stats_fc["data_referencia"] = pd.to_datetime(stats_fc["data_referencia"])

    global_summary = (
        json.loads(GLOBAL_SUMMARY_PATH.read_text(encoding="utf-8"))
        if GLOBAL_SUMMARY_PATH.exists()
        else None
    )
    return stats_bt, stats_summary, stats_fc, global_summary


def build_series_figure(
    bt_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    best_col: str,
    show_full_backtest: bool,
    stats_bt: pd.DataFrame | None = None,
    stats_fc: pd.DataFrame | None = None,
) -> go.Figure:
    """Constrói gráfico de série temporal comparando Real, Baseline D-7, LightGBM e AutoETS."""
    view_bt = bt_df if show_full_backtest else bt_df.tail(84)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view_bt["data_referencia"],
            y=view_bt["y_true"],
            name="Real (ONS SE)",
            line={"color": "#10b981", "width": 2.2},
            hovertemplate="%{y:,.1f} MWmed<extra>Real</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view_bt["data_referencia"],
            y=view_bt["y_pred_baseline"],
            name="Baseline (D-7)",
            line={"color": "#94a3b8", "width": 1.5, "dash": "dot"},
            hovertemplate="%{y:,.1f} MWmed<extra>Baseline D-7</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view_bt["data_referencia"],
            y=view_bt[best_col],
            name="LightGBM (backtest)",
            line={"color": "#0ea5e9", "width": 2.0},
            hovertemplate="%{y:,.1f} MWmed<extra>LightGBM</extra>",
        )
    )

    if stats_bt is not None:
        view_stats = stats_bt[stats_bt["data_referencia"] >= view_bt["data_referencia"].min()]
        fig.add_trace(
            go.Scatter(
                x=view_stats["data_referencia"],
                y=view_stats["y_pred_auto_ets"],
                name="AutoETS (statsforecast)",
                line={"color": "#f59e0b", "width": 1.6, "dash": "dash"},
                hovertemplate="%{y:,.1f} MWmed<extra>AutoETS</extra>",
            )
        )

    fc_start = forecast_df["data_referencia"].min() - pd.Timedelta(hours=12)
    fc_end = forecast_df["data_referencia"].max() + pd.Timedelta(hours=12)
    fig.add_vrect(
        x0=fc_start,
        x1=fc_end,
        fillcolor="rgba(14, 165, 233, 0.08)",
        line_width=0,
        annotation_text="Previsão 7d",
        annotation_position="top left",
        annotation_font={"size": 11, "color": "#0ea5e9"},
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["data_referencia"],
            y=forecast_df["y_pred_lgbm"],
            name="Previsão LightGBM (7d)",
            mode="lines+markers",
            line={"color": "#0ea5e9", "width": 2.6},
            marker={"size": 7},
            hovertemplate="%{y:,.1f} MWmed<extra>Previsão LightGBM</extra>",
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
            hovertemplate="%{y:,.1f} MWmed<extra>Previsão D-7</extra>",
        )
    )
    if stats_fc is not None:
        fig.add_trace(
            go.Scatter(
                x=stats_fc["data_referencia"],
                y=stats_fc["y_pred_auto_ets"],
                name="Previsão AutoETS (7d)",
                mode="lines+markers",
                line={"color": "#f59e0b", "width": 1.8, "dash": "dash"},
                marker={"size": 5},
                hovertemplate="%{y:,.1f} MWmed<extra>Previsão AutoETS</extra>",
            )
        )

    years = sorted({int(pd.Timestamp(d).year) for d in forecast_df["data_referencia"]})
    br_holidays = holidays.country_holidays("BR", years=years)
    for _, row in forecast_df.iterrows():
        dt = pd.Timestamp(row["data_referencia"])
        if dt.date() in br_holidays:
            short_name = "Consciência Negra" if "Consciência" in br_holidays[dt.date()] else "Feriado"
            fig.add_annotation(
                x=dt,
                y=float(row["y_pred_lgbm"]),
                text=f"{dt.strftime('%d/%m')} · {short_name}",
                showarrow=True,
                arrowhead=2,
                ax=-28,
                ay=-36,
                font={"size": 11, "color": "#f59e0b"},
                arrowcolor="#f59e0b",
            )

    fig.update_layout(
        height=390,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        font={"family": "IBM Plex Sans, sans-serif", "size": 12},
        xaxis={"showgrid": False},
        yaxis={
            "title": "Carga (MWmed)",
            "gridcolor": "rgba(148, 163, 184, 0.16)",
            "tickformat": ",.0f",
        },
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
                hovertemplate="%{y:,.1f} MWmed<extra>Baseline D-7</extra>",
            ),
            go.Bar(
                name="LightGBM",
                x=dow_df["dia_semana"],
                y=dow_df[f"{best_name}__mae"],
                marker_color="#0ea5e9",
                hovertemplate="%{y:,.1f} MWmed<extra>LightGBM</extra>",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        height=360,
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        font={"family": "IBM Plex Sans, sans-serif", "size": 12},
        xaxis={"showgrid": False},
        yaxis={
            "title": "MAE (MWmed)",
            "gridcolor": "rgba(148, 163, 184, 0.16)",
            "tickformat": ",.0f",
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig


def main() -> None:
    """Ponto de entrada da interface Streamlit."""
    _inject_typography_polish()

    st.title("Previsão de Carga Diária — Subsistema SE")
    st.caption(
        "Operador Nacional do Sistema (ONS) · Horizonte de 7 dias · "
        "Backtesting temporal de 52 semanas (sem vazamento)"
    )

    missing = [
        p
        for p in (BACKTEST_PRED_PATH, BACKTEST_SUMMARY_PATH, FORECAST_PATH)
        if not p.exists()
    ]
    if missing:
        st.error(
            "Artefatos de avaliação não encontrados. Execute antes: "
            "`uv run python -m src.pipeline --skip-ingest`."
        )
        return

    bt_df, summary, forecast_df = load_dashboard_data()
    stats_bt, stats_summary, stats_fc, global_summary = load_bonus_data()

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
        n_days = int(base_row["n_obs"])
        st.markdown(
            "> **Pergunta do desafio:** *O seu modelo prevê melhor do que simplesmente repetir o "
            "valor do mesmo dia da semana anterior (D-7)? Se sim, quanto melhor?*\n\n"
            f"**Sim.** No backtesting temporal de **52 semanas ({n_days} dias, {dt_min} a {dt_max})**, "
            f"o **LightGBM (`{best_name}`)** superou o D-7 com **MAE de {_fmt_pt(best_mae, 1)} vs "
            f"{_fmt_pt(base_mae, 1)} MWmed** (**-{_fmt_pt(gain_pct, 1)}%** no MAE) e **MAPE de "
            f"{_fmt_pt(best_mape, 2)}% vs {_fmt_pt(base_mape, 2)}%** "
            f"(**-{_fmt_pt(mape_diff_pp, 2)} p.p.**). "
            "Em feriados, vésperas, pós-feriados e ecos de D-7, a redução de MAE chegou a "
            f"**{_fmt_pt(hol_gain_pct, 1)}%**."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "MAE — LightGBM (MWmed)",
        _fmt_pt(best_mae, 1),
        delta=f"-{_fmt_pt(mae_diff, 1)} MW (-{_fmt_pt(gain_pct, 1)}%)",
        delta_color="inverse",
        border=True,
    )
    c2.metric(
        "MAPE — LightGBM",
        f"{_fmt_pt(best_mape, 2)}%",
        delta=f"-{_fmt_pt(mape_diff_pp, 2)} p.p. vs D-7",
        delta_color="inverse",
        border=True,
    )
    c3.metric(
        "MAE — Baseline D-7 (MWmed)",
        _fmt_pt(base_mae, 1),
        delta=f"MAPE {_fmt_pt(base_mape, 2)}%",
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    c4.metric(
        "Feriados e Ecos D-7 (MAE)",
        _fmt_pt(hol_best_mae, 1),
        delta=f"-{_fmt_pt(hol_gain_pct, 1)}% vs D-7 ({_fmt_pt(hol_base_mae, 0)})",
        delta_color="inverse",
        border=True,
    )

    with st.container(border=True):
        st.subheader("Comparação Temporal: Real vs LightGBM vs Baseline D-7")
        ctrl_left, ctrl_right = st.columns([2.2, 1.0])
        with ctrl_left:
            window_option = st.radio(
                "Janela de visualização",
                options=[
                    "Últimas 12 semanas + Próximos 7 dias",
                    "Backtest completo (52 semanas) + Próximos 7 dias",
                ],
                horizontal=True,
                label_visibility="collapsed",
            )
        show_ets = False
        if stats_bt is not None:
            with ctrl_right:
                show_ets = st.checkbox("Sobrepor AutoETS (statsforecast)", value=False)

        fig_series = build_series_figure(
            bt_df,
            forecast_df,
            best_col=best_col,
            show_full_backtest=window_option.startswith("Backtest completo"),
            stats_bt=stats_bt if show_ets else None,
            stats_fc=stats_fc if show_ets else None,
        )
        st.plotly_chart(fig_series, width="stretch", config={"displayModeBar": False})

    combined_rows = list(summary["overall"])
    if stats_summary is not None:
        ets_row = next(r for r in stats_summary["overall"] if r["modelo"] == "statsforecast_auto_ets")
        combined_rows.append(ets_row)
    if global_summary is not None:
        se_global = next(
            (r for r in global_summary["by_subsystem"] if r["id_subsistema"] == "SE"),
            None,
        )
        if se_global is not None:
            combined_rows.append(
                {
                    "modelo": "lightgbm_global_4subs (SE)",
                    "n_obs": se_global["n_obs"],
                    "mae": se_global["global_lgbm__mae"],
                    "mape": se_global["global_lgbm__mape"],
                    "rmse": se_global["global_lgbm__rmse"],
                    "ganho_mae_vs_d7_pct": se_global["ganho_mae_vs_d7_pct"],
                }
            )
    score_df = pd.DataFrame(combined_rows).sort_values("mae").reset_index(drop=True)

    col_left, col_right = st.columns([1.2, 1.0])
    with col_left:
        with st.container(border=True):
            st.subheader("Placar do Backtesting no SE (52 semanas)")
            overall_view = pd.DataFrame(
                {
                    "Modelo": score_df["modelo"],
                    "Dias": score_df["n_obs"].astype(int),
                    "MAE (MWmed)": score_df["mae"].map(lambda v: _fmt_pt(float(v), 1)),
                    "MAPE (%)": score_df["mape"].map(lambda v: f"{_fmt_pt(float(v), 2)}%"),
                    "RMSE (MWmed)": score_df["rmse"].map(lambda v: _fmt_pt(float(v), 1)),
                    "Ganho vs D-7": score_df["ganho_mae_vs_d7_pct"].map(
                        lambda v: f"+{_fmt_pt(float(v), 1)}%"
                    ),
                }
            )
            st.dataframe(overall_view, hide_index=True, width="stretch")

        with st.container(border=True):
            st.subheader("Onde o ganho é maior: Feriados vs Dias Comuns")
            spec_view = pd.DataFrame(
                {
                    "Tipo de Dia": spec_df["tipo_dia"],
                    "Dias": spec_df["n_obs"].astype(int),
                    "Baseline D-7 (MAE)": spec_df["baseline_d7__mae"].map(
                        lambda v: _fmt_pt(float(v), 1)
                    ),
                    "LightGBM (MAE)": spec_df[f"{best_name}__mae"].map(
                        lambda v: _fmt_pt(float(v), 1)
                    ),
                    "Ganho vs D-7": spec_df[f"{best_name}__gain_pct"].map(
                        lambda v: f"+{_fmt_pt(float(v), 1)}%"
                    ),
                }
            )
            st.dataframe(spec_view, hide_index=True, width="stretch")

    with col_right:
        with st.container(border=True):
            st.subheader("MAE por Dia da Semana")
            dow_df = pd.DataFrame(summary["by_weekday"])
            fig_dow = build_weekday_figure(dow_df, best_name)
            st.plotly_chart(fig_dow, width="stretch", config={"displayModeBar": False})

    if global_summary is not None:
        with st.container(border=True):
            st.subheader("Bônus — Modelo Global LightGBM (SE, S, NE, N)")
            glob_df = pd.DataFrame(global_summary["by_subsystem"])
            glob_view = pd.DataFrame(
                {
                    "Subsistema": glob_df["id_subsistema"],
                    "Dias": glob_df["n_obs"].astype(int),
                    "Baseline D-7 (MAE)": glob_df["baseline_d7__mae"].map(
                        lambda v: _fmt_pt(float(v), 1)
                    ),
                    "LightGBM Global (MAE)": glob_df["global_lgbm__mae"].map(
                        lambda v: _fmt_pt(float(v), 1)
                    ),
                    "MAPE Global (%)": glob_df["global_lgbm__mape"].map(
                        lambda v: f"{_fmt_pt(float(v), 2)}%"
                    ),
                    "Ganho vs D-7": glob_df["ganho_mae_vs_d7_pct"].map(
                        lambda v: f"+{_fmt_pt(float(v), 1)}%"
                    ),
                }
            )
            st.dataframe(glob_view, hide_index=True, width="stretch")

    with st.container(border=True):
        st.subheader("Previsão para os Próximos 7 Dias")
        years = sorted({int(pd.Timestamp(d).year) for d in forecast_df["data_referencia"]})
        br_holidays = holidays.country_holidays("BR", years=years)
        diff_series = forecast_df["y_pred_lgbm"] - forecast_df["y_pred_baseline"]
        forecast_view = pd.DataFrame(
            {
                "Data": forecast_df["data_referencia"].dt.strftime("%d/%m/%Y"),
                "Dia da Semana": forecast_df["data_referencia"].dt.day_of_week.map(WEEKDAY_PT),
                "Calendário": [
                    f"Feriado: {br_holidays[d.date()]}" if d.date() in br_holidays else "Dia comum"
                    for d in forecast_df["data_referencia"]
                ],
                "Baseline D-7 (MWmed)": forecast_df["y_pred_baseline"].map(
                    lambda v: _fmt_pt(float(v), 1)
                ),
                "LightGBM (MWmed)": forecast_df["y_pred_lgbm"].map(
                    lambda v: _fmt_pt(float(v), 1)
                ),
                "Diferença (MWmed)": diff_series.map(
                    lambda v: f"{'+' if v >= 0 else ''}{_fmt_pt(float(v), 1)}"
                ),
            }
        )
        st.dataframe(forecast_view, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
