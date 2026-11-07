# What will this flat rent for?

Weekly asking rent, with and without a car space, for all 65,777 flats in the City of Sydney area. Built by **Team HanSolo** for the Microburbs Hackathon (7 November 2026, Task 4: Rents).

| Held-out check (Jun to Jul 2026 adverts) | Result |
|---|---|
| Typical miss (median absolute error) | **$48 a week** |
| Within $100 | **78.7%** |
| R squared | **0.891** |
| Benchmark: Microburbs rental valuation | $86 a week, typically $61 low |

Full method, insights and validation: [`report.html`](report.html).
**Interactive Rent Explorer:** download [`explorer.html`](explorer.html) and open it in a browser. Search any address or click a building to see rent with and without parking, an 80% likely range, the evidence behind the estimate, and the yield on a car space.
Submission file: [`rents_hansolo.csv`](rents_hansolo.csv) (`gnaf_id, rent_with_car_space, rent_without_car_space`).

## Who it helps and who would pay

- **Landlords and investors:** price a listing right the first time, and know whether a car space is worth paying for.
- **Tenants:** check whether an asking rent is fair for the building and street.
- **Agents:** a one-second rental appraisal, the hook that wins property management mandates.
- **Lenders and brokers:** rental income with a confidence range for serviceability checks.
- **Developers and build-to-rent:** what a car space is worth, by suburb and building.

## The data

- Supplied hackathon files: input flats (65,777 rows), past rental adverts (112,457 rows, to 31 July 2026), mesh block home counts 2021 and 2026.
- Nothing dated after 31 July 2026 is used. No live listings.

## The clever step

Old adverts are the best evidence for a flat's rent, but they are stale: carrying a flat's last rent forward underestimates its next advert by about $50 a week. We build a **like-for-like rent index** from rent changes *within the same building and bedroom count*, so new luxury stock cannot distort it. Every past advert is re-priced to the scoring window, then the most local evidence wins (own flat, building, mesh block, suburb, city). A LightGBM model with median loss, trained on rolling time windows that never see the future, corrects what the medians miss.

**What makes this version different**
- **Like-for-like drift index:** removes the low bias that comes from leaning on stale adverts.
- **Honest ranges:** every flat gets an 80% likely range, checked against later adverts and calibrated.
- **Parking priced as an investment:** weekly premium, annual income and gross yield on a car space.
- **Investor yield:** our current rents divided by Microburbs median unit prices give gross yield by suburb, shown in the report and the Explorer.
- **Explainable evidence:** each estimate says whether it came from the flat's own history, its building, or the area.

Insights found on the way:
- Like-for-like asking rents rose about 12% since July 2023.
- A car space is a scarcity premium: worth a lot for 2-bedroom flats in older harbourside suburbs, close to nothing in newer, dense precincts.
- Within the same building, the top floors ask only about 2% more than the lowest floors.
- Gross yields for flats run from about 2.6% (Barangaroo) to 6.4% (Haymarket). The high-yield suburbs are the same dense precincts where a car space adds almost nothing.

## How to run

```bash
pip3 install pandas numpy lightgbm matplotlib
# put the supplied CSVs in data/ (not committed)
python3 src/rents_baseline.py   # drift-adjusted local medians (about 10 s)
python3 src/rents_model.py      # + LightGBM correction, writes rents_hansolo.csv (about 1 min)
python3 src/make_report.py      # writes report.html
python3 src/make_explorer.py    # writes explorer.html (Leaflet bundled in src/vendor)
```

`src/fetch_yield.py` pulls suburb median unit prices (history cut at 31 July 2026) and published yields from the Microburbs API into `cache/` (gitignored). These feed the yield insight only, never the rent estimates.
`src/fetch_api.py` pulls suburb-level data from the Microburbs API (key in `.env` as `MICROBURBS_API_KEY`, never committed).

## Repo layout

```
rents_hansolo.csv     submission
report.html           method, insights, held-out check
src/rents_baseline.py rent index, parking premium, local medians
src/rents_model.py    stacked LightGBM correction and final CSV
src/make_report.py    report and charts
src/make_explorer.py  interactive Rent Explorer
explorer.html         self-contained map and search UI
src/fetch_api.py      Microburbs API helper
```

## Source credits

- Microburbs hackathon data and API (rental adverts, suburb unit prices and yields), © Microburbs. Estimates are automated and indicative only.
- Addresses: [Open G-NAF](https://data.gov.au/data/dataset/19432f89-dc3a-4ef3-b943-5326ef1dbecc), © Geoscape Australia, licensed by the Commonwealth of Australia under the Open G-NAF End User Licence Agreement.
- Block counts: [ABS Census 2021](https://www.abs.gov.au/website-privacy-copyright-and-disclaimer), © Commonwealth of Australia, CC BY 4.0. 2026 block figures are Microburbs estimates, not Census results.
- Map tiles: © OpenStreetMap contributors, © CARTO. Map library: Leaflet (BSD-2-Clause, licence in src/vendor).
- Libraries: pandas, NumPy, LightGBM, Matplotlib.

Not financial advice.
