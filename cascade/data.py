"""Load the four sources and align them on one daily calendar (Europe/Athens).

Sources
-------
ilarionas.csv          PPC  - Ilarionas reservoir, daily natural inflow, 2015-2024
weather_ilarionas.csv  Open-Meteo ERA5 reanalysis (observed weather), 2015-2024
entsoe_cascade.parquet ENTSO-E - hourly generation per unit, 2017 -> 2026-09
entsoe_market.parquet  ENTSO-E - day-ahead price, load forecast, 2017 -> 2026-09
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
UNITS = ["ILARIONAS", "POLYFYTO", "SFIKIA", "ASOMATA"]


def load_inflow() -> pd.DataFrame:
    d = pd.read_csv(DATA / "ilarionas.csv", parse_dates=["date"]).set_index("date")
    out = pd.DataFrame(index=d.index)
    out["inflow"] = d["avg_natural_inflow_m3s"]                 # m3/s, daily mean
    out["res_level"] = d["reservoir_water_level_m"]
    out["turbined_water"] = d["hpp_production_water_1000m3"]
    return out.asfreq("D")


def load_weather() -> pd.DataFrame:
    w = pd.read_csv(DATA / "weather_ilarionas.csv", parse_dates=["date"]).set_index("date")
    keep = ["total_precipitation", "rainfall", "snowfall", "temperature_2m",
            "temperature_min", "temperature_max", "snow_depth", "solar_radiation",
            "degree_days", "api", "snow_water_equivalent"]
    return w[keep].rename(columns={"total_precipitation": "precip",
                                   "temperature_2m": "temp",
                                   "temperature_min": "tmin",
                                   "temperature_max": "tmax",
                                   "snow_water_equivalent": "swe"}).asfreq("D")


def load_generation() -> pd.DataFrame:
    """Daily energy (MWh) per unit from hourly ENTSO-E MW.

    A day is kept only if at least 22 hourly values are present (DST days have
    23/25); the sum is rescaled to the number of hours in that local day.
    """
    g = pd.read_parquet(DATA / "entsoe_cascade.parquet")
    g.columns = [c[0] for c in g.columns]
    g = g[UNITS]
    day = g.index.tz_convert("Europe/Athens").date
    n_hours = g.groupby(day).size()
    present = g.notna().groupby(day).sum()
    total = g.groupby(day).sum(min_count=1)
    daily = total.div(present).mul(n_hours, axis=0)            # rescale partial days
    daily = daily.where(present >= 22)
    daily.index = pd.to_datetime(daily.index)
    daily.columns = [c.lower() for c in daily.columns]
    return daily.asfreq("D")


def load_market() -> pd.DataFrame:
    m = pd.read_parquet(DATA / "entsoe_market.parquet")
    day = m.index.tz_convert("Europe/Athens").date
    out = pd.DataFrame({
        "price": m["price_da"].groupby(day).mean(),
        "price_peak": m["price_da"].groupby(day).max(),
        "load_fc": m["load_forecast_da"].groupby(day).mean(),
        "res_wind_solar": (m["ws_solar"] + m["ws_wind_onshore"]).groupby(day).mean(),
    })
    out.index = pd.to_datetime(out.index)
    return out.asfreq("D")


def load_all() -> pd.DataFrame:
    """One daily frame, 2015-01-01 -> last ENTSO-E day."""
    frames = [load_inflow(), load_weather(), load_generation(), load_market()]
    start = pd.Timestamp("2015-01-01")
    end = max(f.index.max() for f in frames)
    idx = pd.date_range(start, end, freq="D")
    df = pd.concat([f.reindex(idx) for f in frames], axis=1)
    df.index.name = "date"
    # the 2022 price spike: a ratio to the trailing year keeps trees in range
    df["price_rel"] = df["price"] / df["price"].rolling(365, min_periods=60).median()
    return df


if __name__ == "__main__":
    df = load_all()
    print(df.shape, df.index.min().date(), df.index.max().date())
    print(df.notna().groupby(df.index.year).mean().round(2).T)
    print(df.describe().T.round(1))
