"""Forecasters. Every one returns the same long frame:

    date | y_true | y_pred | origin | step  [| q10 | q90]

so models are comparable and can be combined on (origin, step).
"""
import warnings
from typing import Callable, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from cascade import progress as P

H = 7
SARIMA_BATCH = 50          # origins per progress unit
SARIMA_FIT_UNITS = 3       # the one-off parameter fit, in progress units
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.06, n_jobs=1, num_leaves=31,
                   min_child_samples=20, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=42)


def _long(origins, h, y_true, y_pred, **extra) -> pd.DataFrame:
    out = pd.DataFrame({"date": origins + pd.Timedelta(days=h), "y_true": y_true,
                        "y_pred": y_pred, "origin": origins, "step": h})
    for k, v in extra.items():
        out[k] = v
    return out


def origins_for(index, test_start, test_end):
    """Origins whose 7-day window touches the test span; edges are trimmed later."""
    lo = pd.Timestamp(test_start) - pd.Timedelta(days=H)
    hi = pd.Timestamp(test_end) - pd.Timedelta(days=1)
    return index[(index >= lo) & (index <= hi)]


def trim(frame, test_start, test_end):
    m = (frame["date"] >= pd.Timestamp(test_start)) & (frame["date"] <= pd.Timestamp(test_end))
    return frame[m & frame["y_true"].notna()].reset_index(drop=True)


# ------------------------------------------------------------------ direct LGBM
def direct_lgbm(X: pd.DataFrame, y: pd.Series, train_end, pred_origins,
                horizon_feats: Optional[Callable[[int], pd.DataFrame]] = None,
                log_target=False, quantiles=(0.1, 0.9), train_mask=None,
                return_models=False):
    """One LightGBM per lead time h. Training origins satisfy t0+h <= train_end.

    `train_mask` (bool Series on origins) further restricts training rows, used
    for year-block cross-fitting.
    """
    frames, models = [], {}
    for h in range(1, H + 1):
        Xh = X if horizon_feats is None else pd.concat([X, horizon_feats(h)], axis=1)
        yh = y.shift(-h)
        tr = (Xh.index + pd.Timedelta(days=h) <= pd.Timestamp(train_end)) & yh.notna().values
        if train_mask is not None:
            tr &= train_mask.reindex(Xh.index, fill_value=False).values
        yt = np.log1p(yh[tr]) if log_target else yh[tr]
        inv = np.expm1 if log_target else (lambda v: v)

        m = LGBMRegressor(objective="regression_l1", **LGBM_PARAMS).fit(Xh[tr], yt)
        P.tick(f"h={h}")
        Xp = Xh.loc[pred_origins]
        pred = inv(m.predict(Xp))
        extra = {}
        if quantiles:
            lo, hi = _conformal_interval(Xh[tr], yt, quantiles, Xp, h)
            extra = {f"q{int(quantiles[0] * 100)}": np.maximum(inv(lo), 0),
                     f"q{int(quantiles[1] * 100)}": np.maximum(inv(hi), 0)}
        if log_target:
            pred = np.maximum(pred, 0)
        frames.append(_long(pred_origins, h, yh.loc[pred_origins].values, pred, **extra))
        models[h] = (m, list(Xh.columns))
    out = pd.concat(frames, ignore_index=True)
    return (out, models) if return_models else out


CALIB_DAYS = 365


def _conformal_interval(Xtr, ytr, quantiles, Xp, h):
    """Conformalised quantile regression (Romano et al., 2019).

    Raw LightGBM quantiles fit the training data too tightly: a nominal 80%
    band covered ~55-65% out of sample. The last CALIB_DAYS of training are
    held out, the band's miss on them sets a widening margin, and that margin
    is applied to quantile models refitted on all training rows.
    """
    qlo, qhi = quantiles
    cut = ytr.index.max() - pd.Timedelta(days=CALIB_DAYS)
    fit, cal = ytr.index <= cut, ytr.index > cut
    q = lambda a, X, y: LGBMRegressor(objective="quantile", alpha=a, **LGBM_PARAMS).fit(X, y)
    lo_c, hi_c = q(qlo, Xtr[fit], ytr[fit]), q(qhi, Xtr[fit], ytr[fit])
    P.tick(f"h={h} calibration")
    yc = ytr[cal].values
    scores = np.maximum(lo_c.predict(Xtr[cal]) - yc, yc - hi_c.predict(Xtr[cal]))
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (qhi - qlo)) / n)
    margin = np.quantile(scores, level)
    lo_m, hi_m = q(qlo, Xtr, ytr), q(qhi, Xtr, ytr)
    P.tick(f"h={h} quantiles")
    lo, hi = lo_m.predict(Xp) - margin, hi_m.predict(Xp) + margin
    return np.minimum(lo, hi), np.maximum(lo, hi)


# ------------------------------------------------------------------ baselines
def persistence(y: pd.Series, pred_origins) -> pd.DataFrame:
    return pd.concat([_long(pred_origins, h, y.shift(-h).loc[pred_origins].values,
                            y.loc[pred_origins].values) for h in range(1, H + 1)],
                     ignore_index=True)


def seasonal_naive(y: pd.Series, pred_origins, season=7) -> pd.DataFrame:
    """Value on the same weekday one week before the target."""
    return pd.concat([_long(pred_origins, h, y.shift(-h).loc[pred_origins].values,
                            y.shift(season - h).loc[pred_origins].values)
                      for h in range(1, H + 1)], ignore_index=True)


def climatology(y: pd.Series, train_end, pred_origins, log=True) -> pd.DataFrame:
    """Day-of-year median of the training years, smoothed over +/-15 days."""
    tr = y[y.index <= pd.Timestamp(train_end)]
    v = np.log1p(tr) if log else tr
    by_doy = v.groupby(v.index.dayofyear).median().reindex(range(1, 367)).interpolate()
    ext = pd.concat([by_doy.iloc[-15:], by_doy, by_doy.iloc[:15]])
    smooth = ext.rolling(31, center=True).mean().iloc[15:-15]
    smooth = np.expm1(smooth) if log else smooth
    frames = []
    for h in range(1, H + 1):
        tgt = pred_origins + pd.Timedelta(days=h)
        frames.append(_long(pred_origins, h, y.shift(-h).loc[pred_origins].values,
                            smooth.loc[tgt.dayofyear].values))
    return pd.concat(frames, ignore_index=True)


def sarima(y: pd.Series, train_end, pred_origins, order=(2, 0, 1),
           seasonal_order=(1, 0, 1, 7)) -> pd.DataFrame:
    """Parameters fitted once on the training span, then filtered forward with
    fixed parameters; each origin gets a dynamic 7-step forecast that uses only
    observations up to that origin."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = y[y.index >= y.first_valid_index()].asfreq("D")
    tr = y[y.index <= pd.Timestamp(train_end)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(tr, order=order, seasonal_order=seasonal_order, trend="c").fit(disp=False, maxiter=200)
        full = SARIMAX(y, order=order, seasonal_order=seasonal_order, trend="c").filter(fit.params)
    P.tick("fit", units=SARIMA_FIT_UNITS)
    rows = []
    for n, t0 in enumerate(pred_origins, 1):
        if n % SARIMA_BATCH == 0:
            P.tick(f"origin {n}/{len(pred_origins)}")
        i = y.index.get_loc(t0) + 1
        if i + H > len(y):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pr = full.get_prediction(start=i, end=i + H - 1, dynamic=True)
        ci = pr.conf_int(alpha=0.2)
        for h in range(1, H + 1):
            d = y.index[i + h - 1]
            rows.append((d, y.iloc[i + h - 1], pr.predicted_mean.iloc[h - 1], t0, h,
                         ci.iloc[h - 1, 0], ci.iloc[h - 1, 1]))
    out = pd.DataFrame(rows, columns=["date", "y_true", "y_pred", "origin", "step", "q10", "q90"])
    out[["y_pred", "q10", "q90"]] = out[["y_pred", "q10", "q90"]].clip(lower=0)
    return out


def combine(frames: list) -> pd.DataFrame:
    """Equal-weight mean of point forecasts on (origin, step)."""
    keys = ["date", "origin", "step"]
    base = frames[0][keys + ["y_true"]]
    preds = [f.set_index(keys)["y_pred"] for f in frames]
    mean = pd.concat(preds, axis=1).mean(axis=1, skipna=False).rename("y_pred")
    return base.join(mean, on=keys).dropna(subset=["y_pred"]).reset_index(drop=True)


# ------------------------------------------------------------------ metrics
def _metrics(g: pd.DataFrame) -> pd.Series:
    e = g["y_pred"] - g["y_true"]
    yt, yp = g["y_true"], g["y_pred"]
    sst = ((yt - yt.mean()) ** 2).sum()
    r = np.corrcoef(yt, yp)[0, 1] if yp.std() > 0 else np.nan
    kge = 1 - np.sqrt((r - 1) ** 2 + (yp.std() / yt.std() - 1) ** 2 + (yp.mean() / yt.mean() - 1) ** 2)
    out = {"n": len(g), "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
           "R2": 1 - (e ** 2).sum() / sst, "bias": e.mean(), "KGE": kge}
    if "q10" in g and g["q10"].notna().all():
        out["cover80"] = ((yt >= g["q10"]) & (yt <= g["q90"])).mean()
        out["width80"] = (g["q90"] - g["q10"]).mean()
    return pd.Series(out)


def score(preds: dict, reference: str) -> tuple:
    """Overall and per-step metrics; MAE skill relative to `reference`."""
    # score every model on the same (date, step) set
    keys = None
    for f in preds.values():
        k = set(zip(f["date"], f["step"]))
        keys = k if keys is None else keys & k
    rows, by_step = [], []
    for name, f in preds.items():
        f = f[[(d, s) in keys for d, s in zip(f["date"], f["step"])]]
        rows.append(_metrics(f).rename(name))
        s = f.groupby("step").apply(_metrics, include_groups=False)
        s["model"] = name
        by_step.append(s.reset_index())
    overall = pd.DataFrame(rows)
    overall["MAE_skill"] = 1 - overall["MAE"] / overall.loc[reference, "MAE"]
    by_step = pd.concat(by_step, ignore_index=True)
    ref = by_step[by_step["model"] == reference].set_index("step")["MAE"]
    by_step["MAE_skill"] = 1 - by_step["MAE"] / by_step["step"].map(ref)
    return overall, by_step
