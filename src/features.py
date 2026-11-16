"""Engenharia de atributos sem vazamento para previsão de carga no horizonte de 7 dias."""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from src.baseline import LAG_DAYS, validate_daily_continuity

REGIME_MAP: dict[str, int] = {
    "pre_2021_03": 0,
    "nao_despachadas_2021_2023": 1,
    "com_mmgd_pos_2023_04": 2,
}

FEATURE_COLUMNS: tuple[str, ...] = (
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "lag_364",
    "roll_mean_7_lag7",
    "roll_std_7_lag7",
    "roll_min_7_lag7",
    "roll_max_7_lag7",
    "roll_mean_28_lag7",
    "roll_std_28_lag7",
    "ewm_mean_7_lag7",
    "diff_lag7_lag14",
    "ratio_lag7_roll28",
    "day_of_week",
    "is_weekend",
    "is_monday",
    "month",
    "day_of_year",
    "week_of_year",
    "sin_doy",
    "cos_doy",
    "sin_dow",
    "cos_dow",
    "is_holiday",
    "is_holiday_eve",
    "is_post_holiday",
    "is_bridge_day",
    "lag_7_is_holiday",
    "lag_14_is_holiday",
    "regime_code",
)


def _extend_future_horizon(ordered: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Anexa `horizon` dias futuros após a última data observada com target nulo."""
    if horizon <= 0:
        return ordered.assign(is_forecast=False)
    if horizon > LAG_DAYS:
        raise ValueError(f"Sem recursividade, o horizonte máximo permitido é {LAG_DAYS} dias.")

    last_date = ordered["data_referencia"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )
    regime = (
        str(ordered["regime_metodologico"].iloc[-1])
        if "regime_metodologico" in ordered.columns
        else "com_mmgd_pos_2023_04"
    )
    future_df = pd.DataFrame(
        {
            "data_referencia": future_dates,
            "id_subsistema": "SE",
            "nom_subsistema": "Sudeste/Centro-Oeste",
            "carga_mwmed": np.nan,
            "regime_metodologico": regime,
            "is_forecast": True,
        }
    )
    return pd.concat([ordered.assign(is_forecast=False), future_df], ignore_index=True)


def _build_holiday_dates(dates: pd.Series) -> set[pd.Timestamp]:
    """Retorna o conjunto de feriados nacionais no Brasil cobrindo o intervalo das datas."""
    min_year = int(dates.min().year) - 1
    max_year = int(dates.max().year) + 1
    br_holidays = holidays.country_holidays("BR", years=range(min_year, max_year + 1))
    return {pd.Timestamp(dt) for dt in br_holidays}


def build_features(df: pd.DataFrame, horizon: int = 0) -> pd.DataFrame:
    """Constrói matriz de atributos estritamente com informações disponíveis até `d - 7`.

    Quando `horizon > 0` (1..7), gera também as linhas futuras com `carga_mwmed = NaN`
    e todas as features populadas sem vazamento.
    """
    ordered = validate_daily_continuity(df)
    full = _extend_future_horizon(ordered, horizon=horizon)

    y = full["carga_mwmed"].astype(float)
    base_lag = y.shift(LAG_DAYS)

    full["lag_7"] = base_lag
    full["lag_14"] = y.shift(14)
    full["lag_21"] = y.shift(21)
    full["lag_28"] = y.shift(28)
    full["lag_364"] = y.shift(364)

    full["roll_mean_7_lag7"] = base_lag.rolling(window=7, min_periods=1).mean()
    full["roll_std_7_lag7"] = base_lag.rolling(window=7, min_periods=2).std().fillna(0.0)
    full["roll_min_7_lag7"] = base_lag.rolling(window=7, min_periods=1).min()
    full["roll_max_7_lag7"] = base_lag.rolling(window=7, min_periods=1).max()
    full["roll_mean_28_lag7"] = base_lag.rolling(window=28, min_periods=7).mean()
    full["roll_std_28_lag7"] = base_lag.rolling(window=28, min_periods=7).std().fillna(0.0)
    full["ewm_mean_7_lag7"] = base_lag.ewm(alpha=0.3, adjust=False, min_periods=1).mean()

    full["diff_lag7_lag14"] = full["lag_7"] - full["lag_14"]
    full["ratio_lag7_roll28"] = full["lag_7"] / full["roll_mean_28_lag7"]

    dt = full["data_referencia"]
    dow = dt.dt.day_of_week
    doy = dt.dt.day_of_year
    full["day_of_week"] = dow
    full["is_weekend"] = dow.isin([5, 6]).astype(int)
    full["is_monday"] = (dow == 0).astype(int)
    full["month"] = dt.dt.month
    full["day_of_year"] = doy
    full["week_of_year"] = dt.dt.isocalendar().week.astype(int)

    full["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.25)
    full["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.25)
    full["sin_dow"] = np.sin(2.0 * np.pi * dow / 7.0)
    full["cos_dow"] = np.cos(2.0 * np.pi * dow / 7.0)

    holiday_set = _build_holiday_dates(dt)
    is_hol = dt.isin(holiday_set)
    is_hol_tomorrow = (dt + pd.Timedelta(days=1)).isin(holiday_set)
    is_hol_yesterday = (dt - pd.Timedelta(days=1)).isin(holiday_set)

    full["is_holiday"] = is_hol.astype(int)
    full["is_holiday_eve"] = is_hol_tomorrow.astype(int)
    full["is_post_holiday"] = is_hol_yesterday.astype(int)
    full["is_bridge_day"] = (
        ((dow == 0) & is_hol_tomorrow) | ((dow == 4) & is_hol_yesterday)
    ).astype(int)
    full["lag_7_is_holiday"] = (dt - pd.Timedelta(days=7)).isin(holiday_set).astype(int)
    full["lag_14_is_holiday"] = (dt - pd.Timedelta(days=14)).isin(holiday_set).astype(int)

    regime_series = (
        full["regime_metodologico"]
        if "regime_metodologico" in full.columns
        else pd.Series("com_mmgd_pos_2023_04", index=full.index)
    )
    full["regime_code"] = regime_series.map(REGIME_MAP).fillna(2).astype(int)

    return full
