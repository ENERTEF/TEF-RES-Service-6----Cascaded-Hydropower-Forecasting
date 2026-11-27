"""The service: issue one 7-day cascade forecast from a given origin.

    python forecast.py                       # latest origin with full data
    python forecast.py --origin 2024-06-15
    python forecast.py --origin 2026-09-03   # after 2024: public data only

Trains on everything dated <= origin, then forecasts origin+1 ... origin+7:
  1. Ilarionas natural inflow (m3/s) with an 80% interval
  2. Asomata production (MWh/day) with an 80% interval, using (1) as input

When the origin is past the end of the PPC inflow record (2024-12-31), step 1
is skipped and step 2 runs on ENTSO-E data only.
"""
import argparse
from pathlib import Path

import pandas as pd

from cascade import features as F
from cascade import models as M
from cascade.data import load_all
from run import inflow_crossfit

OUT = Path(__file__).resolve().parent / "output" / "forecasts"


def forecast(df: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    df = df.loc[:origin]
    # extend the index by H days so horizon features exist for the target dates
    idx = pd.date_range(df.index.min(), origin + pd.Timedelta(days=M.H), freq="D")
    df = df.reindex(idx)
    po = pd.DatetimeIndex([origin])
    out = pd.DataFrame({"date": origin + pd.to_timedelta(range(1, M.H + 1), unit="D"),
                        "step": range(1, M.H + 1)})

    has_inflow = pd.notna(df["inflow"].get(origin))
    if has_inflow:
        X1 = F.inflow_origin_features(df)
        cal1 = lambda h: F.target_calendar(X1.index, h)
        f1 = M.direct_lgbm(X1, df["inflow"], origin, po, cal1, log_target=True)
        out[["inflow_m3s", "inflow_q10", "inflow_q90"]] = f1.sort_values("step")[["y_pred", "q10", "q90"]].values
        fc = inflow_crossfit(X1, df["inflow"], cal1, origin - pd.Timedelta(days=1))
        fc.loc[origin] = out["inflow_m3s"].values

    g = F.production_origin_features(df)
    X2 = pd.concat([g[F.ASOMATA_OWN], g[F.MARKET], g[F.UPSTREAM]], axis=1)
    extra = None
    if has_inflow:
        X2 = pd.concat([X2, g[F.INFLOW_OBS]], axis=1)
        fcr = fc.reindex(idx)
        extra = lambda h: pd.DataFrame({"inflow_fc_h": fcr[h], "inflow_fc_week": fcr.mean(axis=1)}, index=idx)

    def hf(h):
        parts = [F.target_calendar(idx, h), F.production_horizon_own(df, h)]
        return pd.concat(parts + ([extra(h)] if extra else []), axis=1)

    f2 = M.direct_lgbm(X2, df["asomata"], origin, po, hf)
    out[["asomata_mwh", "asomata_q10", "asomata_q90"]] = f2.sort_values("step")[["y_pred", "q10", "q90"]].values
    out["config"] = "cascade (PPC inflow + ENTSO-E)" if has_inflow else "public only (ENTSO-E)"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", help="YYYY-MM-DD; default: last day with both inflow and production")
    args = ap.parse_args()
    df = load_all()
    if args.origin:
        origin = pd.Timestamp(args.origin)
    else:
        both = df[["inflow", "asomata"]].dropna()
        origin = both.index.max()
    res = forecast(df, origin)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"forecast_{origin:%Y-%m-%d}.csv"
    res.round(2).to_csv(path, index=False)
    print(f"Forecast issued {origin:%Y-%m-%d} · {res['config'].iloc[0]}\n")
    print(res.drop(columns="config").round(1).to_string(index=False))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
