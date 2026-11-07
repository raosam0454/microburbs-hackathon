# Sydney Flat Rents

What will a flat in the City of Sydney rent for, with a car space and without one? This project estimates the weekly asking rent for all 65,777 flats in the area, explains where each number comes from, and shows what it means for anyone renting, letting or buying.

**Try the Rent Explorer:** https://raosam0454.github.io/microburbs-hackathon/explorer.html

Search any address or click a building on the map to see:
- the weekly rent with and without parking
- a likely range, so you know how sure the estimate is
- the evidence behind it (the flat's own history, its building, or the neighbourhood)
- what a car space earns as an investment, and the typical rental yield in that suburb

The full write-up, with charts and checks, is in [`report.html`](report.html). The estimates for every flat are in [`rents_hansolo.csv`](rents_hansolo.csv).

## How accurate is it?

We tested the method the way it will be used: we built estimates from adverts up to 31 May 2026, then compared them with real adverts from June and July 2026.

| | Our estimate | Microburbs rental valuation |
|---|---|---|
| Typical miss | **$48 a week** | $86 a week |
| Within $100 of the asking rent | **78.7%** | 55.2% |
| Share of rent differences explained (R²) | **0.891** | not published |

## Who it's for

- **Tenants** can check whether an asking rent is fair for the building and street.
- **Landlords** can price a listing right the first time instead of guessing and waiting.
- **Investors** can compare yields across suburbs and see whether a car space is worth paying extra for.
- **Agents** get an instant rental appraisal, the conversation that usually wins a property management client.
- **Lenders and brokers** get rental income with a range attached, for borrowing assessments.
- **Developers** can see how much tenants actually pay for parking before deciding how many spaces to build.

## What we learned about Sydney rents

- **Old adverts undersell today's market.** Taking a flat's last advertised rent at face value misses its next advert by about $50 a week, because that advert is typically almost two years old. Like-for-like asking rents have risen about 12% since July 2023.
- **Parking is worth what it is scarce.** For a 2-bedroom flat, a car space adds over $200 a week in Darlinghurst and Elizabeth Bay, but close to nothing in Waterloo, Zetland or Haymarket, where most flats already come with one.
- **Height matters less than people expect.** In the same building, flats on the top floors ask only about 2% more than those near the ground. The building and the street set the rent.
- **The best yields are in the busiest precincts.** Gross yields for flats run from about 2.6% in Barangaroo to 6.4% in Haymarket, and the high-yield suburbs are the same places where parking adds the least.

## How it works

1. **Bring every old advert up to date.** We measure how rents have moved by comparing adverts for the same type of flat in the same building over time, so a wave of new luxury apartments can't distort the trend. Each past advert is then adjusted to August 2026 prices.
2. **Separate the flat from the parking.** We work out what a car space adds in each suburb, for each bedroom count, and set it aside.
3. **Use the closest evidence available.** The flat's own history comes first, then its building, then its block, suburb and the wider city.
4. **Correct what simple averages miss.** A machine learning model (LightGBM) learns where those averages are consistently off, using floor level, building size and age, nearby new construction and location. It only ever learns from adverts that came before the period it is checking.
5. **Add the parking back** to give a rent with and without a car space, plus a likely range calibrated against real adverts.

Every estimate uses only information available by 31 July 2026.

## Where the data comes from

- Microburbs hackathon files: the 65,777 flats, 112,457 past rental adverts up to 31 July 2026, and home counts by block for 2021 and 2026.
- Microburbs API: suburb median unit prices (up to 31 July 2026) and published yields, used only for the yield comparison and not for the rent estimates.

## Running it yourself

```bash
pip3 install pandas numpy lightgbm matplotlib
# place the supplied CSV files in data/
python3 src/rents_baseline.py   # quick estimate from local medians (about 10 seconds)
python3 src/rents_model.py      # full model, writes rents_hansolo.csv (about 1 minute)
python3 src/make_report.py      # builds report.html
python3 src/make_explorer.py    # builds explorer.html
```

To refresh the suburb prices, add your Microburbs key to a `.env` file as `MICROBURBS_API_KEY=...` and run `python3 src/fetch_yield.py`. The key and downloaded data stay on your machine and are never committed.

## What's in this repo

```
explorer.html          interactive map and address search
report.html            findings, method and accuracy checks
rents_hansolo.csv      weekly rent estimates for every flat
src/rents_baseline.py  rent trend, parking value and local medians
src/rents_model.py     machine learning correction and final estimates
src/make_report.py     report and charts
src/make_explorer.py   Rent Explorer page
src/fetch_yield.py     suburb prices and yields from the Microburbs API
src/fetch_api.py       general Microburbs API helper
src/vendor/            Leaflet map library
```

## Credits

- Rental adverts, suburb prices and yields: Microburbs. Estimates are automated and indicative only.
- Addresses: [Open G-NAF](https://data.gov.au/data/dataset/19432f89-dc3a-4ef3-b943-5326ef1dbecc), © Geoscape Australia, licensed by the Commonwealth of Australia under the Open G-NAF End User Licence Agreement.
- Block counts: [ABS Census 2021](https://www.abs.gov.au/website-privacy-copyright-and-disclaimer), © Commonwealth of Australia, CC BY 4.0. The 2026 figures are Microburbs estimates, not Census results.
- Maps: © OpenStreetMap contributors, © CARTO. Map library: Leaflet (BSD-2-Clause).
- Built with pandas, NumPy, LightGBM and Matplotlib.

This is not financial advice.
