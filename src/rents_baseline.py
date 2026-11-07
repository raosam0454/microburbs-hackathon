"""Rents baseline.

1. Re-price every past advert to the scoring window with a market-drift index
   (controls for building x bedroom mix), so old adverts become current evidence.
2. Estimate a parking premium per suburb x bedrooms, shrunk to the city figure.
3. Take the most local evidence available: own flat -> building -> mesh block
   -> suburb -> city, shrinking thin buildings toward the area.

Validation mimics scoring: train on adverts up to 31 May 2026,
test on June-July 2026 adverts (a two-month gap, like Aug-Sep after 31 July).
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
TEAM = "hansolo"
CUTOFF = pd.Timestamp("2026-07-31")


def load():
    F = pd.read_csv(DATA / "rents-input-flats-city-of-sydney.csv")
    L = pd.read_csv(DATA / "rents-past-leases-city-of-sydney.csv", parse_dates=["listing_date"])
    L = L[L["listing_date"] <= CUTOFF].copy()
    # fill missing advert attributes from the flat's own record
    fb = F.set_index("gnaf_id")
    L["bedrooms"] = L["bedrooms"].fillna(L["gnaf_id"].map(fb["bedrooms"]))
    for df in (F, L):
        key = df["latitude"].round(4).astype(str) + "," + df["longitude"].round(4).astype(str)
        df["bkey"] = df["building_address"].fillna(key) + "|" + df["suburb"]
        df["beds"] = df["bedrooms"].fillna(1).clip(0, 4).astype(int)
    L["car"] = L["car_space"].where(L["car_space"].notna(), np.nan)
    L["month"] = L["listing_date"].dt.to_period("M")
    return F, L


def rent_index(L):
    """Monthly log-rent index from within building x bedroom variation."""
    d = L[["bkey", "beds", "month", "weekly_rent"]].copy()
    d["logr"] = np.log(d["weekly_rent"])
    g = d["bkey"] + "|" + d["beds"].astype(str)
    d = d[g.map(g.value_counts()) >= 2]
    g = d["bkey"] + "|" + d["beds"].astype(str)
    d["resid"] = d["logr"] - d.groupby(g)["logr"].transform("mean")
    idx = d.groupby("month")["resid"].median().sort_index()
    full = idx.reindex(pd.period_range(idx.index.min(), idx.index.max(), freq="M"))
    return full.interpolate().ffill().bfill().rolling(3, min_periods=1, center=True).mean()


def project(idx, target):
    """Extend the index to the target month with the last-12-month trend."""
    last = idx.iloc[-12:]
    slope = np.polyfit(np.arange(len(last)), last.values, 1)[0]
    ahead = (target.to_period("M") - idx.index[-1]).n
    return float(idx.iloc[-3:].mean() + slope * ahead), float(slope)


def premiums(L):
    d = L.dropna(subset=["car"])
    def tab(keys):
        g = d.groupby(keys + ["car"])["adj"].agg(["median", "size"]).unstack("car")
        return pd.DataFrame({"prem": g[("median", 1.0)] - g[("median", 0.0)],
                             "n": np.minimum(g[("size", 1.0)], g[("size", 0.0)])})
    city = tab(["beds"])["prem"].clip(0, 300)
    sub = tab(["suburb", "beds"]).reset_index()
    sub["city"] = sub["beds"].map(city)
    k = 30
    n = sub["n"].fillna(0)
    sub["prem"] = ((sub["prem"].fillna(0) * n + sub["city"] * k) / (n + k)).clip(0, 400)
    return sub.set_index(["suburb", "beds"])["prem"], city


def fit_predict(L, T, target_date):
    L = L.copy()
    idx = rent_index(L)
    it, slope = project(idx, target_date)
    L["adj"] = L["weekly_rent"] * np.exp(it - L["month"].map(idx).fillna(idx.iloc[-1]))
    prem_sub, city = premiums(L)
    pfun = lambda s, b: prem_sub.get((s, b), city.get(b, 60.0))
    L["prem"] = [pfun(s, b) for s, b in zip(L["suburb"], L["beds"])]
    L["base"] = L["adj"] - L["prem"] * L["car"].fillna(0.5)

    T = T.copy()
    own = L.sort_values("listing_date").groupby("gnaf_id")["base"].apply(lambda s: s.iloc[-2:].mean())
    b = L.groupby(["bkey", "beds"])["base"].agg(["median", "size"])
    mb = L.groupby(["mb", "beds"])["base"].median()
    sb = L.groupby(["suburb", "beds"])["base"].median()
    cb = L.groupby("beds")["base"].median()

    def look(s, keys):
        return pd.Series([s.get(k, np.nan) for k in keys], index=T.index, dtype=float)
    bk = list(zip(T["bkey"], T["beds"]))
    bm, bn = look(b["median"], bk), look(b["size"], bk).fillna(0)
    area = (look(mb, list(zip(T["mb"], T["beds"])))
            .fillna(look(sb, list(zip(T["suburb"], T["beds"]))))
            .fillna(T["beds"].map(cb)))
    bld = ((bm * bn + area * 3) / (bn + 3)).where(bm.notna(), area)
    o = T["gnaf_id"].map(own)
    base = (0.7 * o + 0.3 * bld).where(o.notna(), bld)
    prem = pd.Series([pfun(s, x) for s, x in zip(T["suburb"], T["beds"])], index=T.index)
    T["rent_without_car_space"] = (base / 5).round() * 5
    T["rent_with_car_space"] = ((base + prem) / 5).round() * 5
    T["source"] = np.select([o.notna(), bm.notna()], ["own_flat", "building"], "area")
    T["prem"] = prem
    return T, {"index_target": it, "monthly_slope": slope, "idx": idx}


def score(P):
    est = np.where(P["car"] == 1, P["rent_with_car_space"], P["rent_without_car_space"])
    y = P["weekly_rent"].values
    err = np.abs(est - y)
    r2 = 1 - np.sum((est - y) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"n": len(P), "median_abs_err": float(np.median(err)), "within_100": float(np.mean(err <= 100)),
            "median_bias": float(np.median(est - y)), "r2": float(r2)}, err


def validate(L, split=pd.Timestamp("2026-05-31")):
    tr = L[L["listing_date"] <= split]
    te = L[(L["listing_date"] > split)].dropna(subset=["car"])
    P, info = fit_predict(tr, te, pd.Timestamp("2026-06-30"))
    s, err = score(P)
    P["abs_err"] = err
    return s, P, info


def main():
    F, L = load()
    s, P, _ = validate(L)
    print("VALIDATION", {k: round(v, 3) for k, v in s.items()})
    print(P.groupby("source")["abs_err"].agg(["median", "size"]).round(0).to_string())
    out, info = fit_predict(L, F, pd.Timestamp("2026-08-31"))
    out = out[["gnaf_id", "rent_with_car_space", "rent_without_car_space"]]
    assert len(out) == len(F) and out["gnaf_id"].is_unique and out.notna().all().all()
    assert (out["rent_with_car_space"] >= out["rent_without_car_space"]).all()
    OUT.mkdir(exist_ok=True)
    out.to_csv(ROOT / f"rents_{TEAM}.csv", index=False, encoding="utf-8")
    print("WROTE", f"rents_{TEAM}.csv", len(out), "rows | monthly drift",
          round(info["monthly_slope"] * 100, 2), "%")


if __name__ == "__main__":
    main()
