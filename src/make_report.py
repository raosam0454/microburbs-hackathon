"""Build report.html from the model outputs (run after rents_model.py)."""
import base64, io, json, pathlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from rents_baseline import load, rent_index, premiums, ROOT

OUT = ROOT / "outputs"
BLUE, GREY, RED = "#1f4e79", "#9aa5b1", "#c0392b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def png(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    met = json.load(open(OUT / "metrics.json"))
    val = pd.read_pickle(OUT / "validation.pkl")
    F, L = load()
    idx = rent_index(L)

    # 1. market drift and the naive miss
    grow12 = (np.exp(idx.iloc[-1] - idx.iloc[-13]) - 1) * 100
    grow3y = (np.exp(idx.iloc[-1] - idx[pd.Period("2023-07", "M")]) - 1) * 100
    tr, te = L[L.listing_date <= "2026-05-31"], L[L.listing_date > "2026-05-31"]
    last = tr.sort_values("listing_date").groupby("gnaf_id")[["weekly_rent", "listing_date"]].last()
    nv = te.join(last, on="gnaf_id", rsuffix="_prev").dropna(subset=["weekly_rent_prev"])
    naive_bias = float(np.median(nv.weekly_rent_prev - nv.weekly_rent))
    naive_age = float(((nv.listing_date - nv.listing_date_prev).dt.days / 365).median())

    s = idx[idx.index >= pd.Period("2019-01", "M")]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(s.index.to_timestamp(), (np.exp(s - s.iloc[-1]) * 100), color=BLUE)
    ax.axhline(100, color=GREY, lw=0.8, ls=":")
    ax.set_ylabel("Rent level (Jul 2026 = 100)")
    ax.set_title("Same flats, same buildings: asking rents over time", loc="left")
    chart_idx = png(fig)

    # 2. parking premium by suburb (2 bedrooms)
    it = idx.iloc[-1]
    L["adj"] = L.weekly_rent * np.exp(it - L.month.map(idx))
    ps, city = premiums(L)
    p2 = ps.xs(2, level="beds")
    cnt = L.dropna(subset=["car"]).query("beds == 2").groupby(["suburb", "car"]).size().unstack().min(axis=1)
    p2 = p2[cnt.reindex(p2.index).fillna(0) >= 30].sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(p2.index, p2.values, color=BLUE)
    ax.set_xlabel("Extra weekly rent for a car space, 2-bedroom flats ($)")
    ax.set_title("Parking is worth most where it is scarce", loc="left")
    chart_park = png(fig)
    p2.index = p2.index.str.replace(r" \(.*\)", "", regex=True)
    top, bottom = p2.sort_values(ascending=False).head(3), p2.sort_values().head(3)

    # 3. floor effect within the same building and bedroom count
    L["floor"] = L.gnaf_id.map(F.set_index("gnaf_id")["floor_level"])
    g = L.bkey + "|" + L.beds.astype(str)
    L["res"] = np.log(L.adj) - L.groupby(g)["adj"].transform(lambda x: np.log(x).mean())
    d = L.dropna(subset=["floor"])
    d = d[g.loc[d.index].map(g.value_counts()) >= 5]
    d["fb"] = pd.cut(d.floor, [0, 2, 5, 10, 20, 40, 400], labels=["1-2", "3-5", "6-10", "11-20", "21-40", "41+"])
    fe = (np.exp(d.groupby("fb", observed=True)["res"].mean()) - 1) * 100
    floor_gap = float(fe.iloc[-1] - fe.iloc[0])

    # 4. validation error bands and sources
    bands = pd.cut(val.abs_err, [-1, 50, 100, 200, 1e9],
                   labels=["within $50", "$50 to $100", "$100 to $200", "over $200"]).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.bar(bands.index.astype(str), bands.values, color=[BLUE, BLUE, GREY, RED])
    for i, v in enumerate(bands.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_title("How far our estimates were from later adverts (Jun to Jul 2026)", loc="left")
    chart_err = png(fig)
    src = val.groupby("source")["abs_err"].agg(["median", "size"])
    src_rows = "".join(f"<tr><td>{k.replace('_', ' ')}</td><td>{int(r['size']):,}</td><td>${r['median']:.0f}</td></tr>"
                       for k, r in src.sort_values("size", ascending=False).iterrows())
    imp = met["importance"]

    b, m = met["baseline"], met["model"]
    n_train_adverts = int((L.listing_date <= "2026-05-31").sum())
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>What will this flat rent for? Team HanSolo</title>
<style>
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:780px;margin:40px auto;padding:0 18px;color:#1d2733;line-height:1.55}}
h1{{color:{BLUE};font-size:28px;margin-bottom:4px}} h2{{color:{BLUE};margin-top:36px;font-size:20px}}
.sub{{color:#5b6875;margin-top:0}} .cards{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}}
.card{{flex:1;min-width:160px;background:#f3f6fa;border-radius:10px;padding:14px}}
.card b{{display:block;font-size:26px;color:{BLUE}}} img{{width:100%}}
table{{border-collapse:collapse;width:100%;font-size:14px}} td,th{{border-bottom:1px solid #dde3ea;padding:6px;text-align:left}}
.flow{{font-family:Menlo,monospace;background:#f3f6fa;padding:12px;border-radius:8px;font-size:13px;white-space:pre}}
.small{{color:#5b6875;font-size:13px}}
</style></head><body>
<h1>What will this flat rent for?</h1>
<p class="sub">Team HanSolo. Weekly rent with and without a car space for 65,777 City of Sydney flats, using only information available by 31 July 2026.</p>

<h2>What we found along the way</h2>
<div class="cards">
<div class="card"><b>${-naive_bias:.0f} too low</b>Using a flat's last advertised rent as-is underestimates the next advert by ${-naive_bias:.0f} a week. Its last advert is typically {naive_age:.1f} years old.</div>
<div class="card"><b>+{grow3y:.0f}%</b>Rise in like-for-like asking rents since July 2023 (+{grow12:.1f}% in the last 12 months). Old adverts must be re-priced before use.</div>
<div class="card"><b>${top.iloc[0]:.0f} vs ${bottom.iloc[0]:.0f}</b>Weekly value of a car space for a 2-bedroom flat in {top.index[0]} versus {bottom.index[0]}.</div>
</div>
<img src="{chart_idx}" alt="Rent index">
<p><b>Parking is a scarcity premium, not a fixed price.</b> In older inner-east and harbourside areas, where street parking is scarce, a car space adds the most:
{", ".join(f"{k} ${v:.0f}" for k, v in top.items())} a week for 2-bedroom flats. In denser, newer precincts where most flats come with a space,
it adds little: {", ".join(f"{k} ${v:.0f}" for k, v in bottom.items())}. For an investor, ${top.iloc[0]:.0f} a week is ${top.iloc[0] * 52:,.0f} a year:
a space bought for $100,000 would return about {top.iloc[0] * 52 / 1000:.0f}% gross. The same space in {bottom.index[0]} returns close to nothing.</p>
<img src="{chart_park}" alt="Parking premium by suburb">
<p class="small">Suburbs with at least 30 adverts both with and without parking. Premiums compare drift-adjusted asking rents and are shrunk toward the city figure where data is thin.</p>
<p><b>Height matters less than people think.</b> Within the same building and bedroom count, flats on floors 41+ ask only {floor_gap:.1f}% more than floors 1 to 2. The building and the street set the rent far more than the floor does.</p>

<h2>What our separate check showed</h2>
<p>R squared is the share of the differences in weekly rent that our estimates explain (1.0 would be perfect).</p>
<div class="cards">
<div class="card"><b>${m['median_abs_err']:.0f} a week</b>Typical miss on {m['n']:,} later adverts</div>
<div class="card"><b>{m['within_100'] * 100:.1f}%</b>Within $100 ({int(round(m['within_100'] * m['n'])):,} of {m['n']:,})</div>
<div class="card"><b>{m['r2']:.3f}</b>R squared</div>
</div>
<p>For reference, the task page lists a typical miss of $86 for the Microburbs rental valuation and $100 for a simple estimate.</p>
<img src="{chart_err}" alt="Error bands">

<h2>What data supported the result?</h2>
<p>The supplied input file (65,777 flats in 33 suburbs), {len(L):,} past rental adverts from {L.listing_date.min():%B %Y} to 31 July 2026, and the
supplied mesh block home counts for 2021 and 2026. No advert dated after 31 July 2026 was used. The 4,246 addresses outside the supplied City files
are estimated from their building, mesh block and suburb neighbours using the coordinates already in the input file. No live listings were used.</p>

<h2>How did we estimate rent?</h2>
<div class="flow">past adverts --> re-price to Aug 2026 with a like-for-like rent index
             --> remove parking premium (suburb x bedrooms)
             --> most local evidence: own flat > building > mesh block > suburb > city
             --> LightGBM correction (floor, building size and age, new supply, location)
             --> add parking premium back: with and without car space</div>
<p><b>The clever step is the rent index.</b> It is measured only from rent changes within the same building and bedroom count, so it is not fooled by
new luxury stock entering the market. Every old advert is re-priced to the scoring window and projected forward with the last 12 months' trend
({met['monthly_drift_pct']:.2f}% a month). This removes the low bias that comes from leaning on old adverts.</p>
<p>A LightGBM model (median loss, to match the median dollar score) then learns where that local estimate is systematically off. Its training rows
come from rolling two-month windows since June 2024: for each window the estimate is built only from earlier adverts, exactly like the real task
({met['train_rows']:,} rows). Biggest drivers: gap between the flat's own history and the area ({imp['own_est_gap'] * 100:.0f}%), the base estimate
({imp['base_est'] * 100:.0f}%), and how old the flat's last advert is ({imp['months_since_own'] * 100:.0f}%). Estimates are rounded to $5 and the
with-parking figure is never below the without-parking figure.</p>

<h2>How did we check it?</h2>
<p>We split adverts by date. Everything up to 31 May 2026 built the estimates; adverts from 1 June to 31 July 2026 ({m['n']:,} with known parking)
were held out and only used for scoring, mimicking the two-month gap between 31 July and the real Aug to Sep test. The parking column used is the
one matching each advert.</p>
<table><tr><th>Evidence used</th><th>Adverts</th><th>Typical miss</th></tr>{src_rows}</table>
<table style="margin-top:12px"><tr><th>Method</th><th>Typical miss</th><th>Within $100</th><th>R squared</th></tr>
<tr><td>Local medians with rent index</td><td>${b['median_abs_err']:.0f}</td><td>{b['within_100'] * 100:.1f}%</td><td>{b['r2']:.3f}</td></tr>
<tr><td>+ LightGBM correction (submitted)</td><td>${m['median_abs_err']:.0f}</td><td>{m['within_100'] * 100:.1f}%</td><td>{m['r2']:.3f}</td></tr></table>

<h2>What remains uncertain?</h2>
<p>Our check ran about ${m['median_bias']:.0f} high in June to July, so the forward projection may be slightly optimistic if the market cooled. Flats
with no advert history and few neighbours (area-only estimates) miss far more. Parking premiums compare different flats, so part of each premium may
reflect bigger or better flats that tend to have a space. We do not model aspect, views or condition. The official result on Aug to Sep adverts is
the real test.</p>
<p class="small">Data: Microburbs hackathon files and API. Addresses: Open G-NAF, Geoscape Australia. Block counts: ABS Census 2021, CC BY 4.0 (2026 figures are Microburbs estimates). Not financial advice.</p>
</body></html>"""
    (ROOT / "report.html").write_text(html, encoding="utf-8")
    print("WROTE report.html |", {"grow3y": round(grow3y, 1), "naive_bias": naive_bias, "floor_gap": round(floor_gap, 1)})


if __name__ == "__main__":
    main()
