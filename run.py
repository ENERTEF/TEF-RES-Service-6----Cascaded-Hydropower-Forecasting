"""Run the full cascade backtest: Track 1 (Ilarionas inflow) -> Track 2 (Asomata production).

    python run.py            # everything, then figures
    python run.py --no-sarima

Writes output/results/*.csv, output/results/preds_*.parquet, output/figures/*.png.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cascade import features as F
from cascade import models as M
from cascade import progress as P
from cascade.data import load_all

OUT = Path(__file__).resolve().parent / "output"
RES, FIG = OUT / "results", OUT / "figures"

# Split A: both tracks and the cascade link. PPC inflow ends 2024-12-31.
A = dict(train_end="2022-12-31", test_start="2023-01-01", test_end="2024-12-31")
# Split B: public data only (ENTSO-E), the configuration that can run today.
B = dict(train_end="2024-12-31", test_start="2025-01-01")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===================================================================== Track 1
def track1(df):
    log("Track 1 - Ilarionas natural inflow")
    d = df.loc[:"2024-12-31"]
    y = d["inflow"]
    X = F.inflow_origin_features(d)
    cal = lambda h: F.target_calendar(X.index, h)
    oracle = lambda h: pd.concat([cal(h), F.future_weather(d, h)], axis=1)
    po = M.origins_for(X.index, A["test_start"], A["test_end"])

    preds = {"Persistence": M.persistence(y, po),
             "Climatology": M.climatology(y, A["train_end"], po)}
    P.stage("Track 1 · LightGBM")
    preds["LightGBM"] = M.direct_lgbm(X, y, A["train_end"], po, cal, log_target=True)
    P.stage("Track 1 · LightGBM + weather (oracle)")
    preds["LightGBM + weather (oracle)"] = M.direct_lgbm(X, y, A["train_end"], po, oracle, log_target=True)
    preds = {k: M.trim(v, A["test_start"], A["test_end"]) for k, v in preds.items()}
    overall, by_step = M.score(preds, reference="Persistence")

    log("Track 1 - cross-fitted inflow forecasts for the cascade link")
    P.stage("Track 1 · cross-fitted inflow forecasts")
    fc = inflow_crossfit(X, y, cal, A["train_end"])
    return preds, overall, by_step, fc


def inflow_crossfit(X, y, cal, train_end):
    """Inflow forecasts for every origin, to feed Track 2 as a feature.

    Origins up to `train_end`: year-block cross-fitting (each year forecast by a
    model trained on the other years), so the production model trains on
    forecasts with realistic error rather than in-sample fits.
    Origins after `train_end`: the model trained through `train_end`.
    """
    train_end = pd.Timestamp(train_end)
    fc = []
    for yr in range(X.index.year.min() + 1, train_end.year + 1):
        lo, hi = pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=8), pd.Timestamp(f"{yr}-12-31")
        mask = pd.Series(~((X.index >= lo) & (X.index <= hi)), index=X.index)
        porig = X.index[(X.index.year == yr) & (X.index <= train_end)]
        fc.append(M.direct_lgbm(X, y, train_end, porig, cal, log_target=True,
                                quantiles=None, train_mask=mask))
    later = X.index[X.index > train_end]
    if len(later):
        fc.append(M.direct_lgbm(X, y, train_end, later, cal, log_target=True, quantiles=None))
    return pd.concat(fc).pivot(index="origin", columns="step", values="y_pred")


# ===================================================================== Track 2
def track2_split(df, split, inflow_fc=None, sarima=True, tag="A"):
    y = df["asomata"]
    g = F.production_origin_features(df)
    idx = df.index
    test_end = split.get("test_end") or str(y.last_valid_index().date())
    po = M.origins_for(idx, split["test_start"], test_end)

    def hf(extra=None):
        def f(h):
            parts = [F.target_calendar(idx, h), F.production_horizon_own(df, h)]
            if extra is not None:
                parts.append(extra(h))
            return pd.concat(parts, axis=1)
        return f

    X_single = pd.concat([g[F.ASOMATA_OWN], g[F.MARKET]], axis=1)
    X_up = pd.concat([X_single, g[F.UPSTREAM]], axis=1)

    preds = {"Persistence": M.persistence(y, po),
             "Seasonal naive (week)": M.seasonal_naive(y, po)}
    if sarima:
        log(f"Track 2/{tag} - SARIMA")
        P.stage(f"Track 2/{tag} · SARIMA")
        preds["SARIMA"] = M.sarima(y.loc["2017-01-01":], split["train_end"], po)
    log(f"Track 2/{tag} - LightGBM configs")
    P.stage(f"Track 2/{tag} · LightGBM - Asomata only")
    preds["LightGBM - Asomata only"] = M.direct_lgbm(X_single, y, split["train_end"], po, hf(), quantiles=None)
    P.stage(f"Track 2/{tag} · LightGBM + upstream units")
    preds["LightGBM + upstream units"] = M.direct_lgbm(X_up, y, split["train_end"], po, hf())

    models = None
    if inflow_fc is not None:
        X_casc = pd.concat([X_up, g[F.INFLOW_OBS]], axis=1)
        fc = inflow_fc.reindex(idx)
        fc_feats = lambda h: pd.DataFrame({"inflow_fc_h": fc[h], "inflow_fc_week": fc.mean(axis=1)}, index=idx)
        q = df["inflow"]
        or_feats = lambda h: pd.DataFrame({"inflow_fc_h": q.shift(-h),
                                           "inflow_fc_week": q.rolling(7).mean().shift(-7)}, index=idx)
        P.stage(f"Track 2/{tag} · Cascade + inflow forecast")
        preds["LightGBM cascade: + inflow forecast"], models = M.direct_lgbm(
            X_casc, y, split["train_end"], po, hf(fc_feats), return_models=True)
        P.stage(f"Track 2/{tag} · Cascade + actual inflow (oracle)")
        preds["LightGBM cascade: + actual inflow (oracle)"] = M.direct_lgbm(
            X_casc, y, split["train_end"], po, hf(or_feats), quantiles=None)
        main = "LightGBM cascade: + inflow forecast"
    else:
        main = "LightGBM + upstream units"
    if sarima:
        preds["Ensemble (SARIMA + " + ("LightGBM cascade" if inflow_fc is not None else "LightGBM upstream") + ")"] = \
            M.combine([preds["SARIMA"], preds[main]])

    preds = {k: M.trim(v, split["test_start"], test_end) for k, v in preds.items()}
    overall, by_step = M.score(preds, reference="Seasonal naive (week)")
    return preds, overall, by_step, models, main


def planned_units(df, sarima=True):
    """Units of work the run will tick through, for the progress ETA."""
    h = M.H
    t1 = 2 * 3 * h + (2022 - 2016 + 1) * h + h            # two quantile configs + cross-fit
    def t2(split, cascade):
        end = split.get("test_end") or str(df["asomata"].last_valid_index().date())
        n = len(M.origins_for(df.index, split["test_start"], end))
        s = (M.SARIMA_FIT_UNITS + n // M.SARIMA_BATCH) if sarima else 0
        return s + h + 3 * h + (3 * h + h if cascade else 0)
    return t1 + t2(A, True) + t2(B, False) + 1


def importance(models):
    rows = []
    for h, (m, cols) in models.items():
        imp = pd.Series(m.booster_.feature_importance("gain"), index=cols)
        rows.append((imp / imp.sum()).rename(h))
    return pd.concat(rows, axis=1)


def save(name, preds, overall, by_step):
    long = pd.concat([v.assign(model=k) for k, v in preds.items()], ignore_index=True)
    long.to_parquet(RES / f"preds_{name}.parquet")
    overall.round(4).to_csv(RES / f"metrics_{name}.csv", index_label="model")
    by_step.round(4).to_csv(RES / f"metrics_by_step_{name}.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(overall.round(3).to_string(), "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sarima", action="store_true")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    df = load_all()
    P.start(planned_units(df, sarima=not args.no_sarima), OUT)
    df.to_parquet(RES / "daily_data.parquet")

    p1, o1, s1, inflow_fc = track1(df)
    save("track1_inflow", p1, o1, s1)
    inflow_fc.to_parquet(RES / "inflow_forecasts_crossfit.parquet")

    p2, o2, s2, mdl, main2 = track2_split(df, A, inflow_fc, sarima=not args.no_sarima, tag="A")
    save("track2_production", p2, o2, s2)
    importance(mdl).to_csv(RES / "importance_cascade.csv")

    p3, o3, s3, _, _ = track2_split(df, B, None, sarima=not args.no_sarima, tag="B")
    save("track2_production_live", p3, o3, s3)

    from cascade import plots
    P.stage("Figures")
    plots.make_all(df, RES, FIG)
    P.tick()
    P.finish()
    log("done")


if __name__ == "__main__":
    main()
