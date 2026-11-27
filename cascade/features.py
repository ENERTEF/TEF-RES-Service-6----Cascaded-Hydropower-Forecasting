"""Origin-frozen feature builders.

Every forecast is issued at the end of an origin day t0 for target days
t0+1 ... t0+H. A feature row indexed by t0 may read only data dated <= t0.

Two kinds of feature frames, both indexed by origin t0:

* origin features   X(t0)       - shared by every lead time
* horizon features  Z(t0, h)    - describe the target day t0+h. Calendar is
                                   always known; anything else here (weather,
                                   inflow on the target day) is either a real
                                   forecast or an explicitly labelled oracle.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd


# --------------------------------------------------------------------- calendar
def _orthodox_easter(year: int) -> date:
    a, b, c = year % 4, year % 7, year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    month = (d + e + 114) // 31
    day = (d + e + 114) % 31 + 1
    return date(year, month, day) + timedelta(days=13)          # Julian -> Gregorian


def greek_holidays(years) -> set:
    out = set()
    for y in years:
        e = _orthodox_easter(y)
        out |= {date(y, 1, 1), date(y, 1, 6), date(y, 3, 25), date(y, 5, 1),
                date(y, 8, 15), date(y, 10, 28), date(y, 12, 25), date(y, 12, 26),
                e - timedelta(48), e - timedelta(2), e, e + timedelta(1), e + timedelta(50)}
    return out


def calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    hol = greek_holidays(range(index.year.min(), index.year.max() + 1))
    doy = index.dayofyear
    return pd.DataFrame({
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "weekday": index.weekday,
        "month": index.month,
        "holiday": [d.date() in hol for d in index],
    }, index=index).astype(float)


def target_calendar(index: pd.DatetimeIndex, h: int) -> pd.DataFrame:
    """Calendar of t0+h, indexed by t0."""
    cal = calendar(index + pd.Timedelta(days=h))
    cal.index = index
    return cal.add_prefix("tgt_")


# ------------------------------------------------------------------ track 1
def inflow_origin_features(df: pd.DataFrame) -> pd.DataFrame:
    q = df["inflow"]
    lq = np.log1p(q)
    X = pd.DataFrame(index=df.index)
    for k in range(7):
        X[f"lq_lag{k}"] = lq.shift(k)
    for w in (3, 7, 14, 30, 90):
        X[f"lq_mean{w}"] = lq.rolling(w, min_periods=1).mean()
    X["lq_std7"] = lq.rolling(7).std()
    X["lq_diff1"] = lq.diff(1)
    X["lq_diff7"] = lq.diff(7)
    X["res_level"] = df["res_level"]
    X["res_level_d1"] = df["res_level"].diff(1)
    X["res_level_d7"] = df["res_level"].diff(7)
    X["turbined_mean7"] = df["turbined_water"].rolling(7).mean()
    for w in (1, 3, 7, 14, 30):
        X[f"precip_sum{w}"] = df["precip"].rolling(w).sum()
    X["snowfall_sum7"] = df["snowfall"].rolling(7).sum()
    X["snowfall_sum30"] = df["snowfall"].rolling(30).sum()
    X["temp_mean3"] = df["temp"].rolling(3).mean()
    X["temp_mean7"] = df["temp"].rolling(7).mean()
    X["tmax_mean3"] = df["tmax"].rolling(3).mean()
    X["swe"] = df["swe"]
    X["swe_d7"] = df["swe"].diff(7)
    X["snow_depth"] = df["snow_depth"]
    X["api"] = df["api"]
    X["degree_days_sum7"] = df["degree_days"].rolling(7).sum()
    return X


def future_weather(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Observed weather over (t0, t0+h] - a PERFECT weather forecast (oracle)."""
    fwd = lambda s, w: s.rolling(w).sum().shift(-w)               # sum over t0+1..t0+w
    fmean = lambda s, w: s.rolling(w).mean().shift(-w)
    return pd.DataFrame({
        "fw_precip_h": df["precip"].shift(-h),
        "fw_precip_cum": fwd(df["precip"], h),
        "fw_snowfall_cum": fwd(df["snowfall"], h),
        "fw_temp_mean": fmean(df["temp"], h),
        "fw_tmax_h": df["tmax"].shift(-h),
        "fw_swe_h": df["swe"].shift(-h),
        "fw_degdays_cum": fwd(df["degree_days"], h),
    }, index=df.index)


# ------------------------------------------------------------------ track 2
ASOMATA_OWN = "own"
UPSTREAM = "upstream"
INFLOW_OBS = "inflow_obs"
MARKET = "market"


def production_origin_features(df: pd.DataFrame) -> dict:
    """Feature groups for Asomata, returned separately so configs can mix them."""
    a = df["asomata"]
    own = pd.DataFrame(index=df.index)
    for k in range(7):
        own[f"aso_lag{k}"] = a.shift(k)
    for w in (7, 14, 30, 90):
        own[f"aso_mean{w}"] = a.rolling(w, min_periods=max(1, w // 2)).mean()
    own["aso_std7"] = a.rolling(7, min_periods=4).std()
    own["aso_zero_days7"] = (a < 1).astype(float).where(a.notna()).rolling(7, min_periods=4).sum()

    up = pd.DataFrame(index=df.index)
    for u in ("ilarionas", "polyfyto", "sfikia"):
        up[f"{u}_lag0"] = df[u]
        up[f"{u}_lag1"] = df[u].shift(1)
        up[f"{u}_mean7"] = df[u].rolling(7, min_periods=4).mean()
        up[f"{u}_mean30"] = df[u].rolling(30, min_periods=15).mean()
    up["upstream_total_mean7"] = up[["ilarionas_mean7", "polyfyto_mean7", "sfikia_mean7"]].sum(axis=1, min_count=1)

    inflow = pd.DataFrame(index=df.index)
    inflow["inflow_lag0"] = df["inflow"]
    inflow["inflow_mean7"] = df["inflow"].rolling(7).mean()
    inflow["inflow_mean30"] = df["inflow"].rolling(30).mean()
    inflow["res_level"] = df["res_level"]
    inflow["res_level_d7"] = df["res_level"].diff(7)

    mk = pd.DataFrame(index=df.index)
    mk["price_lag0"] = df["price"]
    mk["price_rel_lag0"] = df["price_rel"]
    mk["price_peak_lag0"] = df["price_peak"]
    mk["price_rel_mean7"] = df["price_rel"].rolling(7, min_periods=4).mean()
    mk["load_fc_lag0"] = df["load_fc"]
    mk["res_ws_lag0"] = df["res_wind_solar"]
    return {ASOMATA_OWN: own, UPSTREAM: up, INFLOW_OBS: inflow, MARKET: mk}


def production_horizon_own(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Same weekday one and two weeks before the target day (both <= t0 for h <= 7)."""
    a = df["asomata"]
    return pd.DataFrame({"aso_tgt_lag7": a.shift(7 - h),
                         "aso_tgt_lag14": a.shift(14 - h)}, index=df.index)
