"""Rents model: baseline + LightGBM correction (stacked, time-honest).

The baseline (rents_baseline.py) gives a drift-adjusted local estimate.
LightGBM then learns where that estimate is systematically off, using flat
features the medians ignore (floor, building size and age, new supply, etc.).

Training rows for the correction come from rolling two-month windows: for each
window, the baseline is fitted only on earlier adverts and asked to predict the
window, exactly like the real task. So the model never sees the future.
"""
import json
import pathlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from rents_baseline import load, fit_predict, score, ROOT, TEAM

FEATS = ["beds", "bathrooms", "car", "floor_level", "flats_in_building", "new_share",
         "block_homes_2026", "first_year", "ptype", "latitude", "longitude",
         "base_est", "prem", "bld_n", "own_est_gap", "bld_est_gap", "months_since_own", "src"]


def attach(T, F, B):
    """Add flat-level features from the input file and mesh-block file."""
    T = T.copy()
    f = F.set_index("gnaf_id")
    for c in ["floor_level", "flats_in_building", "block_new_homes_2021_2026",
              "block_homes_2026", "first_ever_listed", "property_type"]:
        if c not in T:
            T[c] = T["gnaf_id"].map(f[c])
        else:
            T[c] = T[c].fillna(T["gnaf_id"].map(f[c]))
    T["bathrooms"] = T["bathrooms"].fillna(T["gnaf_id"].map(f["bathrooms"])) if "bathrooms" in T else T["gnaf_id"].map(f["bathrooms"])
    nh = T["block_new_homes_2021_2026"].fillna(T["mb"].map(B["new_homes"]))
    hh = T["block_homes_2026"].fillna(T["mb"].map(B["homes_2026"]))
    T["block_homes_2026"] = hh
    T["new_share"] = nh / hh.replace(0, np.nan)
    T["first_year"] = pd.to_datetime(T["first_ever_listed"], errors="coerce").dt.year
    T["ptype"] = T["property_type"].astype(str).str.lower().map({"apartment": 0, "unit": 1, "studio": 2, "flat": 1})
    T["src"] = T["source"].map({"own_flat": 2, "building": 1, "area": 0})
    T["own_est_gap"] = T["own_est"] - T["base_est"]
    T["bld_est_gap"] = T["bld_est"] - T["base_est"]
    return T


def window_rows(L, F, B, start, end):
    """Baseline fitted on adverts before `start`, predicting adverts in [start, end]."""
    tr = L[L["listing_date"] < start]
    te = L[(L["listing_date"] >= start) & (L["listing_date"] <= end)].dropna(subset=["car"])
    mid = start + (end - start) / 2
    P, _ = fit_predict(tr, te, mid)
    P = attach(P, F, B)
    P["est"] = np.where(P["car"] == 1, P["rent_with_car_space"], P["rent_without_car_space"])
    P["resid"] = P["weekly_rent"] - P["est"]
    return P


def windows(first, last):
    out, s = [], pd.Timestamp(first)
    while s + pd.DateOffset(months=2) - pd.Timedelta(days=1) <= pd.Timestamp(last):
        out.append((s, s + pd.DateOffset(months=2) - pd.Timedelta(days=1)))
        s += pd.DateOffset(months=2)
    return out


def train(rows, alpha=None):
    obj = dict(objective="quantile", alpha=alpha) if alpha else dict(objective="l1")
    params = dict(**obj, learning_rate=0.04, num_leaves=31, min_data_in_leaf=60,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)
    return lgb.train(params, lgb.Dataset(rows[FEATS], rows["resid"]), num_boost_round=400)


def main():
    F, L = load()
    B = pd.read_csv(ROOT / "data" / "blocks-homes-and-people-2021-2026.csv").set_index("mb")

    # ---- validation: correction trained on windows ending 31 May, tested Jun-Jul
    trw = windows("2024-06-01", "2026-05-31")
    rows = pd.concat([window_rows(L, F, B, s, e) for s, e in trw], ignore_index=True)
    test = window_rows(L, F, B, pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-31"))
    m = train(rows)
    base_s, _ = score(test)
    test["rent_with_car_space"] += m.predict(test[FEATS])
    test["rent_without_car_space"] += m.predict(test[FEATS])
    model_s, err = score(test)
    est_v = np.where(test["car"] == 1, test["rent_with_car_space"], test["rent_without_car_space"])
    q_lo, q_hi = train(rows, 0.1), train(rows, 0.9)
    lo_v = est_v + q_lo.predict(test[FEATS]) - m.predict(test[FEATS])
    hi_v = est_v + q_hi.predict(test[FEATS]) - m.predict(test[FEATS])
    coverage = float(np.mean((test["weekly_rent"] >= lo_v) & (test["weekly_rent"] <= hi_v)))
    width = float(np.median(hi_v - lo_v))
    # calibrate width so 80% of held-out adverts fall inside (split-conformal scale)
    y = test["weekly_rent"].values
    ks = np.linspace(0.8, 2.0, 121)
    cov = [np.mean((y >= est_v + k * (lo_v - est_v)) & (y <= est_v + k * (hi_v - est_v))) for k in ks]
    k80 = float(ks[np.argmax(np.array(cov) >= 0.80)])
    print(f"RANGE  calibration factor {k80:.2f}")
    print(f"RANGE  P10-P90 coverage {coverage:.1%} (target 80%) | median width ${width:.0f}")
    print("BASELINE", {k: round(v, 3) for k, v in base_s.items()})
    print("MODEL   ", {k: round(v, 3) for k, v in model_s.items()})

    # ---- final: correction trained on windows up to 31 July, applied to all flats
    rows = pd.concat([rows, test.assign(resid=test["weekly_rent"] - np.where(
        test["car"] == 1, test["rent_with_car_space"] - m.predict(test[FEATS]),
        test["rent_without_car_space"] - m.predict(test[FEATS])))], ignore_index=True)
    m = train(rows)
    P, info = fit_predict(L, F, pd.Timestamp("2026-08-31"))
    P = attach(P, F, B)
    P["car"] = P["car_space"]
    adj = m.predict(P[FEATS])
    q_lo, q_hi = train(rows, 0.1), train(rows, 0.9)
    P["lo_off"] = q_lo.predict(P[FEATS]) - adj
    P["hi_off"] = q_hi.predict(P[FEATS]) - adj
    P["lo_off"] = (k80 * P["lo_off"]).clip(upper=0)
    P["hi_off"] = (k80 * P["hi_off"]).clip(lower=0)
    use_model = model_s["median_abs_err"] < base_s["median_abs_err"]
    if use_model:
        P["rent_without_car_space"] = P["rent_without_car_space"] + adj
        P["rent_with_car_space"] = P["rent_with_car_space"] + adj
    P["rent_without_car_space"] = (P["rent_without_car_space"].clip(lower=150) / 5).round() * 5
    P["rent_with_car_space"] = np.maximum((P["rent_with_car_space"] / 5).round() * 5, P["rent_without_car_space"])
    out = P[["gnaf_id", "rent_with_car_space", "rent_without_car_space"]].copy()
    out[["rent_with_car_space", "rent_without_car_space"]] = out[["rent_with_car_space", "rent_without_car_space"]].astype(int)
    assert len(out) == len(F) and out["gnaf_id"].is_unique and out.notna().all().all()
    assert (out["rent_with_car_space"] >= out["rent_without_car_space"]).all()
    out.to_csv(ROOT / f"rents_{TEAM}.csv", index=False, encoding="utf-8")
    print("WROTE", f"rents_{TEAM}.csv", len(out), "| used model correction:", use_model)

    # artefacts for the report
    rep = ROOT / "outputs"
    rep.mkdir(exist_ok=True)
    imp = pd.Series(m.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
    json.dump({"baseline": base_s, "model": model_s, "range_coverage": coverage, "range_k": k80, "range_width": width, "train_rows": int(len(rows)),
               "monthly_drift_pct": info["monthly_slope"] * 100,
               "importance": (imp / imp.sum()).round(4).to_dict()},
              open(rep / "metrics.json", "w"), indent=2)
    test.assign(abs_err=err).to_pickle(rep / "validation.pkl")
    P.to_pickle(rep / "final_predictions.pkl")
    info["idx"].to_frame("idx").to_pickle(rep / "rent_index.pkl")


if __name__ == "__main__":
    main()
