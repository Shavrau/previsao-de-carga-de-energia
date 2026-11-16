"""Dashboard Streamlit para previsão de carga diária (ONS SE) e comparação com Baseline D-7."""

from __future__ import annotations

import json
from pathlib import Path

import holidays
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

GOLD_PATH = Path("data/gold/carga_energia_se.parquet")
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
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "theme" not in st.session_state:
    st.session_state.theme = "light"


def toggle_theme() -> None:
    """Alterna tema claro/escuro do dashboard."""
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"


IS_DARK = st.session_state.theme == "dark"


def inject_styles() -> None:
    """Injeta CSS do design system minimalista Zinc."""
    bg = "#09090b" if IS_DARK else "#ffffff"
    bg_subtle = "#0c0c0f" if IS_DARK else "#f9fafb"
    card = "#0c0c0f" if IS_DARK else "#ffffff"
    card_hover = "#131316" if IS_DARK else "#f4f4f5"
    border = "#1e1e24" if IS_DARK else "#e4e4e7"
    border_subtle = "#16161a" if IS_DARK else "#f0f0f2"
    text = "#fafafa" if IS_DARK else "#09090b"
    text_dim = "#52525b" if IS_DARK else "#a1a1aa"
    green = "#22c55e" if IS_DARK else "#16a34a"
    green_muted = "rgba(34,197,94,0.12)" if IS_DARK else "rgba(22,163,74,0.08)"
    red = "#ef4444" if IS_DARK else "#dc2626"
    red_muted = "rgba(239,68,68,0.12)" if IS_DARK else "rgba(220,38,38,0.08)"
    amber = "#f59e0b" if IS_DARK else "#d97706"
    amber_muted = "rgba(245,158,11,0.12)" if IS_DARK else "rgba(217,119,6,0.08)"
    shadow = "none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {{
            --bg: {bg};
            --bg-subtle: {bg_subtle};
            --card: {card};
            --card-hover: {card_hover};
            --border: {border};
            --border-subtle: {border_subtle};
            --text: {text};
            --text-muted: #71717a;
            --text-dim: {text_dim};
            --accent: #2563eb;
            --accent-muted: #1d4ed8;
            --green: {green};
            --green-muted: {green_muted};
            --red: {red};
            --red-muted: {red_muted};
            --amber: {amber};
            --amber-muted: {amber_muted};
            --shadow: {shadow};
            --radius: 10px;
        }}

        header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
        [data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
        div[data-testid="stSidebarCollapsedControl"] {{
            display: none !important;
        }}

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main,
        .block-container, section[data-testid="stMain"] {{
            background-color: var(--bg) !important;
            color: var(--text) !important;
            font-family: 'DM Sans', -apple-system, sans-serif !important;
        }}

        .block-container {{
            padding: 1.75rem 2.25rem 3rem !important;
            max-width: 1360px !important;
        }}

        [data-testid="stHorizontalBlock"] {{ gap: 1.25rem !important; }}
        [data-testid="stVerticalBlock"] > div:has(> [data-testid="stHorizontalBlock"]) {{
            margin-bottom: 0.5rem !important;
        }}

        .brand-title {{
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text);
            margin: 0;
        }}
        .brand-sub {{
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }}

        .answer-banner {{
            background: var(--bg-subtle);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            border-radius: var(--radius);
            padding: 1rem 1.25rem;
            margin: 0.75rem 0 1rem;
            font-size: 0.9rem;
            line-height: 1.5;
            color: var(--text);
        }}

        .metric-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.25rem 1.4rem;
            box-shadow: var(--shadow);
            transition: background-color 0.15s ease;
        }}
        .metric-card:hover {{ background: var(--card-hover); }}
        .metric-label {{ font-size: 0.78rem; color: var(--text-muted); font-weight: 500; }}
        .metric-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.65rem;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.03em;
            margin-top: 0.2rem;
        }}
        .metric-delta {{
            font-size: 0.75rem;
            font-weight: 500;
            margin-top: 0.45rem;
            padding: 2px 8px;
            border-radius: 6px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }}
        .delta-up {{ color: var(--green); background: var(--green-muted); }}
        .delta-down {{ color: var(--red); background: var(--red-muted); }}
        .delta-warn {{ color: var(--amber); background: var(--amber-muted); }}

        .chart-wrap {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.2rem 1.2rem 0.6rem;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }}
        .chart-title {{ font-size: 0.88rem; font-weight: 600; color: var(--text); }}
        .chart-subtitle {{ font-size: 0.75rem; color: var(--text-dim); margin-bottom: 0.6rem; }}

        button[data-baseweb="tab"] {{
            background: transparent !important;
            color: var(--text-muted) !important;
            font-size: 0.835rem !important;
            font-weight: 500 !important;
            padding: 0.55rem 1rem !important;
            border: 1px solid transparent !important;
            border-radius: 7px !important;
            transition: all 0.15s ease !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: var(--text) !important;
            background: var(--card) !important;
            border-color: var(--border) !important;
        }}
        [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
            display: none !important;
        }}
        [data-baseweb="tab-list"] {{
            gap: 4px !important;
            background: var(--bg-subtle) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            padding: 3px;
            margin-bottom: 0.75rem;
        }}

        .data-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 0.82rem;
        }}
        .data-table th {{
            text-align: left;
            padding: 0.6rem 0.8rem;
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            border-bottom: 1px solid var(--border);
        }}
        .data-table td {{
            padding: 0.65rem 0.8rem;
            color: var(--text);
            border-bottom: 1px solid var(--border-subtle);
        }}
        .data-table tr:last-child td {{ border-bottom: none; }}
        .mono {{ font-family: 'JetBrains Mono', monospace; }}

        .badge {{
            display: inline-block;
            padding: 2px 9px;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 500;
        }}
        .badge-green {{ color: var(--green); background: var(--green-muted); }}
        .badge-amber {{ color: var(--amber); background: var(--amber-muted); }}
        .badge-blue {{ color: var(--accent); background: rgba(37,99,235,0.1); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plot_layout() -> dict[str, object]:
    """Layout padrão dos gráficos Plotly adaptado ao tema."""
    grid_color = "rgba(255,255,255,0.06)" if IS_DARK else "rgba(0,0,0,0.05)"
    font_color = "#a1a1aa" if IS_DARK else "#71717a"
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "DM Sans, sans-serif", "color": font_color, "size": 12},
        "margin": {"l": 10, "r": 10, "t": 16, "b": 10},
        "xaxis": {"gridcolor": grid_color, "zerolinecolor": grid_color},
        "yaxis": {"gridcolor": grid_color, "zerolinecolor": grid_color},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    }


def metric_card(label: str, value: str, delta: str | None = None, delta_type: str = "up") -> None:
    """Renderiza card de KPI."""
    cls = f"delta-{delta_type}"
    arrow = "↑" if delta_type == "up" else ("↓" if delta_type == "down" else "→")
    delta_html = f'<div class="metric-delta {cls}">{arrow} {delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[dict]], pd.DataFrame]:
    """Carrega dados Gold, backtest e previsão futura."""
    gold_df = pd.read_parquet(GOLD_PATH)
    gold_df["data_referencia"] = pd.to_datetime(gold_df["data_referencia"])

    bt_df = pd.read_parquet(BACKTEST_PRED_PATH)
    bt_df["data_referencia"] = pd.to_datetime(bt_df["data_referencia"])

    summary = json.loads(BACKTEST_SUMMARY_PATH.read_text(encoding="utf-8"))
    forecast_df = pd.read_parquet(FORECAST_PATH)
    forecast_df["data_referencia"] = pd.to_datetime(forecast_df["data_referencia"])
    return gold_df, bt_df, summary, forecast_df


def _format_mw(val: float) -> str:
    return f"{val:,.1f} MW".replace(",", "X").replace(".", ",").replace("X", ".")


def main() -> None:
    """Ponto de entrada da interface Streamlit."""
    inject_styles()

    head_left, head_right = st.columns([8, 1.3])
    with head_left:
        st.markdown(
            """
            <div>
                <h1 class="brand-title">Previsão de Carga Diária — Subsistema SE (ONS)</h1>
                <div class="brand-sub">Comparação temporal entre LightGBM (lags ≥ 7 dias + calendário/feriados) e Baseline ingênuo (D-7)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with head_right:
        theme_label = "☀️ Light" if IS_DARK else "🌙 Dark"
        st.button(theme_label, on_click=toggle_theme, use_container_width=True)

    missing = [p for p in (GOLD_PATH, BACKTEST_PRED_PATH, BACKTEST_SUMMARY_PATH, FORECAST_PATH) if not p.exists()]
    if missing:
        st.error(
            "Artefatos necessários não encontrados. Execute antes: "
            "`uv run python -m src.ingest`, `uv run python -m src.transform`, "
            "`uv run python -m src.train` e `uv run python -m src.evaluate`."
        )
        return

    gold_df, bt_df, summary, forecast_df = load_dashboard_data()
    overall_df = pd.DataFrame(summary["overall"])
    base_row = overall_df[overall_df["modelo"] == "baseline_d7"].iloc[0]
    best_row = overall_df[overall_df["modelo"] != "baseline_d7"].sort_values("mae").iloc[0]
    best_name = str(best_row["modelo"])
    best_col = f"y_pred_{best_name}"

    gain_pct = float(best_row["ganho_mae_vs_d7_pct"])
    mape_diff_pp = float(base_row["mape"]) - float(best_row["mape"])

    st.markdown(
        f"""
        <div class="answer-banner">
            <strong>Resposta ao desafio:</strong> Sim. Em backtesting temporal de 52 semanas ({int(base_row['n_obs'])} dias, {bt_df['data_referencia'].min().strftime('%d/%m/%Y')} a {bt_df['data_referencia'].max().strftime('%d/%m/%Y')}),
            o modelo <strong>LightGBM ({best_name})</strong> superou o baseline D-7 com <strong>MAE de {best_row['mae']:,.1f} MWmed</strong>
            (vs <strong>{base_row['mae']:,.1f} MWmed</strong>) e <strong>MAPE de {best_row['mape']:.2f}%</strong> (vs <strong>{base_row['mape']:.2f}%</strong>) — redução de <strong>{gain_pct:.1f}%</strong> no erro absoluto médio.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            f"MAE LightGBM ({best_name})",
            f"{best_row['mae']:,.1f} MWmed",
            delta=f"{gain_pct:.1f}% menor que D-7",
            delta_type="up",
        )
    with c2:
        metric_card(
            "MAPE LightGBM",
            f"{best_row['mape']:.2f}%",
            delta=f"-{mape_diff_pp:.2f} p.p. vs D-7",
            delta_type="up",
        )
    with c3:
        metric_card(
            "MAE Baseline (D-7)",
            f"{base_row['mae']:,.1f} MWmed",
            delta=f"MAPE {base_row['mape']:.2f}%",
            delta_type="warn",
        )
    with c4:
        fc_mean = float(forecast_df["y_pred_lgbm"].mean())
        base_fc_mean = float(forecast_df["y_pred_baseline"].mean())
        fc_diff = ((fc_mean - base_fc_mean) / base_fc_mean) * 100.0
        metric_card(
            "Média Prevista (Próximos 7d)",
            f"{fc_mean:,.0f} MWmed",
            delta=f"{fc_diff:+.1f}% vs D-7",
            delta_type="up" if fc_diff <= 0 else "warn",
        )

    tab1, tab2, tab3 = st.tabs(
        ["📈 Série & Previsão 7 Dias", "📊 Backtest 52 Semanas", "🧠 Feriados & Metodologia"]
    )

    with tab1:
        window_choice = st.radio(
            "Janela de visualização histórica",
            options=["Últimos 90 dias (Backtest + Previsão)", "Último ano (365 dias)", "Série completa (2018–2026)"],
            horizontal=True,
        )

        fig = go.Figure()
        if window_choice.startswith("Últimos 90"):
            view_bt = bt_df.tail(90)
            fig.add_trace(
                go.Scatter(
                    x=view_bt["data_referencia"],
                    y=view_bt["y_true"],
                    name="Real (ONS SE)",
                    line={"color": "#fafafa" if IS_DARK else "#09090b", "width": 2.4},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=view_bt["data_referencia"],
                    y=view_bt["y_pred_baseline"],
                    name="Baseline (D-7)",
                    line={"color": "#94a3b8", "width": 1.6, "dash": "dash"},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=view_bt["data_referencia"],
                    y=view_bt[best_col],
                    name=f"LightGBM ({best_name})",
                    line={"color": "#2563eb", "width": 2.2},
                )
            )
            subtitle = "Comparação fora da amostra nos últimos 90 dias + projeção dos próximos 7 dias"
        else:
            hist_slice = gold_df.tail(365) if "365" in window_choice else gold_df
            fig.add_trace(
                go.Scatter(
                    x=hist_slice["data_referencia"],
                    y=hist_slice["carga_mwmed"],
                    name="Carga Observada (Gold SE)",
                    line={"color": "#2563eb" if "365" in window_choice else "#64748b", "width": 1.6},
                )
            )
            if "completa" in window_choice:
                fig.add_vline(x="2023-04-29", line_dash="dot", line_color="#f59e0b")
                fig.add_annotation(
                    x="2023-04-29",
                    y=1.02,
                    yref="paper",
                    text="Início MMGD (29/04/2023)",
                    showarrow=False,
                    font={"color": "#f59e0b", "size": 11},
                )
            subtitle = "Série histórica limpa (camada Gold) + previsão dos próximos 7 dias"

        fig.add_trace(
            go.Scatter(
                x=forecast_df["data_referencia"],
                y=forecast_df["y_pred_lgbm"],
                name="Previsão LightGBM (7d)",
                mode="lines+markers",
                line={"color": "#16a34a", "width": 2.5},
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
        fig.update_layout(**plot_layout(), height=410, yaxis_title="Carga (MWmed)")

        st.markdown(
            f"""
            <div class="chart-wrap">
                <div class="chart-title">Carga Diária do Subsistema SE e Previsão para os Próximos 7 Dias</div>
                <div class="chart-subtitle">{subtitle}</div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        br_holidays = holidays.country_holidays("BR", years=[2026])
        rows_html = ""
        for _, row in forecast_df.iterrows():
            dt = pd.Timestamp(row["data_referencia"])
            dow_name = WEEKDAY_PT[int(dt.day_of_week)]
            hol_name = br_holidays.get(dt.date())
            badge = (
                f'<span class="badge badge-amber">Feriado: {hol_name}</span>'
                if hol_name
                else '<span class="badge badge-blue">Dia comum</span>'
            )
            diff_mw = float(row["y_pred_lgbm"]) - float(row["y_pred_baseline"])
            rows_html += (
                f"<tr>"
                f"<td class='mono'>{dt.strftime('%d %b %Y')}</td>"
                f"<td>{dow_name}</td>"
                f"<td>{badge}</td>"
                f"<td class='mono'>{_format_mw(float(row['y_pred_baseline']))}</td>"
                f"<td class='mono'><strong>{_format_mw(float(row['y_pred_lgbm']))}</strong></td>"
                f"<td class='mono'>{diff_mw:+,.1f} MW</td>"
                f"</tr>"
            )

        st.markdown(
            f"""
            <div class="chart-wrap">
                <div class="chart-title">Detalhamento da Previsão — Próximos 7 Dias</div>
                <div class="chart-subtitle">Repare como o LightGBM ajusta automaticamente o feriado de 20/11 (Dia da Consciência Negra)</div>
                <table class="data-table">
                    <thead>
                        <tr><th>Data</th><th>Dia da Semana</th><th>Calendário</th><th>Baseline D-7</th><th>LightGBM</th><th>Diferença</th></tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab2:
        overall_rows = ""
        for idx, row in overall_df.iterrows():
            badge = (
                '<span class="badge badge-green">Vencedor</span>'
                if idx == 0
                else ('<span class="badge badge-amber">Baseline</span>' if row["modelo"] == "baseline_d7" else '<span class="badge badge-blue">Concorrente</span>')
            )
            overall_rows += (
                f"<tr>"
                f"<td class='mono'>{row['modelo']}</td>"
                f"<td>{badge}</td>"
                f"<td class='mono'>{int(row['n_obs'])}</td>"
                f"<td class='mono'>{row['mae']:,.2f} MW</td>"
                f"<td class='mono'>{row['mape']:.3f}%</td>"
                f"<td class='mono'>{row['rmse']:,.2f} MW</td>"
                f"<td class='mono'>{row['ganho_mae_vs_d7_pct']:+.2f}%</td>"
                f"</tr>"
            )
        st.markdown(
            f"""
            <div class="chart-wrap">
                <div class="chart-title">Placar Consolidado — Backtesting Temporal (52 Semanas)</div>
                <div class="chart-subtitle">Validação expanding window em blocos semanais de 7 dias sem vazamento</div>
                <table class="data-table">
                    <thead>
                        <tr><th>Modelo</th><th>Papel</th><th>Dias</th><th>MAE</th><th>MAPE</th><th>RMSE</th><th>Ganho MAE vs D-7</th></tr>
                    </thead>
                    <tbody>{overall_rows}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_l, col_r = st.columns(2)
        with col_l:
            dow_df = pd.DataFrame(summary["by_weekday"])
            fig_dow = go.Figure(
                data=[
                    go.Bar(name="Baseline (D-7)", x=dow_df["dia_semana"], y=dow_df["baseline_d7__mae"], marker_color="#94a3b8"),
                    go.Bar(name=f"LightGBM ({best_name})", x=dow_df["dia_semana"], y=dow_df[f"{best_name}__mae"], marker_color="#2563eb"),
                ]
            )
            fig_dow.update_layout(**plot_layout(), barmode="group", height=320, yaxis_title="MAE (MWmed)")
            st.markdown(
                """
                <div class="chart-wrap">
                    <div class="chart-title">MAE por Dia da Semana</div>
                    <div class="chart-subtitle">Desempenho fora da amostra de Segunda a Domingo</div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_dow, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r:
            hor_df = pd.DataFrame(summary["by_horizon"])
            fig_hor = go.Figure(
                data=[
                    go.Scatter(name="Baseline (D-7)", x=hor_df["horizon_day"], y=hor_df["baseline_d7__mae"], mode="lines+markers", line={"color": "#94a3b8", "dash": "dash"}),
                    go.Scatter(name=f"LightGBM ({best_name})", x=hor_df["horizon_day"], y=hor_df[f"{best_name}__mae"], mode="lines+markers", line={"color": "#2563eb", "width": 2.4}),
                ]
            )
            fig_hor.update_layout(**plot_layout(), height=320, xaxis_title="Horizonte (dias à frente)", yaxis_title="MAE (MWmed)")
            st.markdown(
                """
                <div class="chart-wrap">
                    <div class="chart-title">MAE por Dia do Horizonte (h = 1..7)</div>
                    <div class="chart-subtitle">Consistência ao longo da semana prevista usando apenas lags ≥ 7 dias</div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_hor, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        mon_df = pd.DataFrame(summary["by_month"])
        fig_mon = go.Figure(
            data=[
                go.Scatter(name="Baseline (D-7)", x=mon_df["mes"], y=mon_df["baseline_d7__mae"], mode="lines+markers", line={"color": "#94a3b8", "dash": "dash"}),
                go.Scatter(name=f"LightGBM ({best_name})", x=mon_df["mes"], y=mon_df[f"{best_name}__mae"], mode="lines+markers", line={"color": "#2563eb", "width": 2.4}),
            ]
        )
        fig_mon.update_layout(**plot_layout(), height=310, yaxis_title="MAE Mensal (MWmed)")
        st.markdown(
            """
            <div class="chart-wrap">
                <div class="chart-title">Evolução Mensal do MAE ao Longo das 52 Semanas</div>
                <div class="chart-subtitle">Estabilidade sazonal entre verão, outono, inverno e primavera</div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_mon, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        spec_df = pd.DataFrame(summary["by_special_day"])
        spec_rows = ""
        for _, row in spec_df.iterrows():
            gain = float(row[f"{best_name}__gain_pct"])
            spec_rows += (
                f"<tr>"
                f"<td>{row['tipo_dia']}</td>"
                f"<td class='mono'>{int(row['n_obs'])}</td>"
                f"<td class='mono'>{row['baseline_d7__mae']:,.2f} MW</td>"
                f"<td class='mono'>{row[f'{best_name}__mae']:,.2f} MW</td>"
                f"<td class='mono'><span class='badge badge-green'>{gain:+.2f}%</span></td>"
                f"</tr>"
            )
        st.markdown(
            f"""
            <div class="chart-wrap">
                <div class="chart-title">Impacto em Feriados, Vésperas, Pós-Feriados e Eco de D-7</div>
                <div class="chart-subtitle">Onde o D-7 ingênuo sofre mais e como as flags de calendário reduzem o erro</div>
                <table class="data-table">
                    <thead>
                        <tr><th>Tipo de Dia</th><th>Dias</th><th>MAE Baseline (D-7)</th><th>MAE LightGBM</th><th>Ganho vs D-7</th></tr>
                    </thead>
                    <tbody>{spec_rows}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="chart-wrap">
                <div class="chart-title">Decisões Técnicas e Rigor Anti-Vazamento</div>
                <div class="chart-subtitle">Resumo arquitetural para apresentação</div>
                <table class="data-table">
                    <thead><tr><th>Dimensão</th><th>Decisão Aplicada</th></tr></thead>
                    <tbody>
                        <tr><td><strong>Sem Vazamento (7d)</strong></td><td>Todas as features autorregressivas usam apenas observações ≤ <code>d - 7</code> (<code>lag_7</code>..<code>lag_364</code> e médias/desvios sobre <code>shift(7)</code>).</td></tr>
                        <tr><td><strong>Quebra Metodológica ONS</strong></td><td>Inclusão da MMGD em <code>29/04/2023</code> elevou a carga média em ~7.000 MWmed; treinar na janela <code>post_mmgd</code> superou o histórico completo.</td></tr>
                        <tr><td><strong>Backtesting</strong></td><td>52 folds semanais consecutivos (expanding window), nunca usando split aleatório.</td></tr>
                        <tr><td><strong>Rastreabilidade</strong></td><td>Ingestão DuckDB <code>httpfs</code> → dbt Silver/Gold (17 testes) → MLflow em Docker (WSL2).</td></tr>
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
