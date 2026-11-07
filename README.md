# What will this flat rent for?

Weekly asking rent, with and without a car space, for all 65,777 flats in the City of Sydney area. Built by **Team HanSolo** for the Microburbs Hackathon (7 November 2026, Task 4: Rents).

| Held-out check (Jun to Jul 2026 adverts) | Result |
|---|---|
| Typical miss (median absolute error) | **$48 a week** |
| Within $100 | **78.7%** |
| R squared | **0.891** |
| Benchmark: Microburbs rental valuation | $86 a week, typically $61 low |

Full method, insights and validation: [`report.html`](report.html).
Submission file: [`rents_hansolo.csv`](rents_hansolo.csv) (`gnaf_id, rent_with_car_space, rent_without_car_space`).

## Who it helps

- **Landlords and investors:** price a listing right the first time, and know whether a car space is worth paying for.
- **Tenants:** check whether an asking rent is fair for the building and street.
- **Agents and lenders:** a fast, explainable rental appraisal for any flat, including ones never advertised.

## The data

- Supplied hackathon files: input flats (65,777 rows), past rental adverts (112,457 rows, to 31 July 2026), mesh block home counts 2021 and 2026.
- Nothing dated after 31 July 2026 is used. No live listings.

## The clever step

Old adverts are the best evidence for a flat's rent, but they are stale: carrying a flat's last rent forward underestimates its next advert by about $50 a week. We build a **like-for-like rent index** from rent changes *within the same building and bedroom count*, so new luxury stock cannot distort it. Every past advert is re-priced to the scoring window, then the most local evidence wins (own flat, building, mesh block, suburb, city). A LightGBM model with median loss, trained on rolling time windows that never see the future, corrects what the medians miss.

Insights found on the way:
- Like-for-like asking rents rose about 12% since July 2023.
- A car space is a scarcity premium: worth a lot for 2-bedroom flats in older harbourside suburbs, close to nothing in newer, dense precincts.
- Within the same building, the top floors ask only about 2% more than the lowest floors.

## How to run

```bash
pip3 install pandas numpy lightgbm matplotlib
# put the supplied CSVs in data/ (not committed)
python3 src/rents_baseline.py   # drift-adjusted local medians (about 10 s)
python3 src/rents_model.py      # + LightGBM correction, writes rents_hansolo.csv (about 1 min)
python3 src/make_report.py      # writes report.html
```

`src/fetch_api.py` pulls suburb-level data from the Microburbs API (key in `.env` as `MICROBURBS_API_KEY`, never committed).

## Repo layout

```
rents_hansolo.csv     submission
report.html           method, insights, held-out check
src/rents_baseline.py rent index, parking premium, local medians
src/rents_model.py    stacked LightGBM correction and final CSV
src/make_report.py    report and charts
src/fetch_api.py      Microburbs API helper
```

## Source credits

- Microburbs hackathon data and API, © Microburbs. Estimates are automated and indicative only.
- Addresses: [Open G-NAF](https://data.gov.au/data/dataset/19432f89-dc3a-4ef3-b943-5326ef1dbecc), © Geoscape Australia, licensed by the Commonwealth of Australia under the Open G-NAF End User Licence Agreement.
- Block counts: [ABS Census 2021](https://www.abs.gov.au/website-privacy-copyright-and-disclaimer), © Commonwealth of Australia, CC BY 4.0. 2026 block figures are Microburbs estimates, not Census results.
- Libraries: pandas, NumPy, LightGBM, Matplotlib.

Not financial advice.
