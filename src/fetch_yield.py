"""Fetch suburb unit prices and published yields from the Microburbs API.

Run on a machine that can reach the API:   python3 src/fetch_yield.py
Reads MICROBURBS_API_KEY from .env. 58 calls (2 per suburb), about 10 seconds.
Writes cache/suburb_prices.json (gitignored: raw API output is not committed).
Prices are taken from the history series at or before 31 July 2026.
"""
import json, pathlib, ssl, time, urllib.parse, urllib.request, urllib.error
try:  # macOS python.org builds often lack root certificates
    import certifi
    CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = ssl.create_default_context()
SUBURBS = ["Alexandria", "Barangaroo", "Beaconsfield (NSW)", "Camperdown (NSW)", "Centennial Park (NSW)",
           "Chippendale", "Darlinghurst", "Darlington (Sydney - NSW)", "Dawes Point", "Elizabeth Bay",
           "Erskineville", "Eveleigh", "Forest Lodge", "Glebe (NSW)", "Haymarket", "Millers Point",
           "Moore Park", "Potts Point", "Pyrmont", "Redfern", "Rosebery (NSW)", "Rushcutters Bay",
           "Surry Hills", "Sydney", "The Rocks (Sydney - NSW)", "Ultimo", "Waterloo (NSW)",
           "Woolloomooloo", "Zetland"]

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE, CUTOFF = "https://api-ht.microburbs.com.au", "2026-07-31"
KEY = next(l.split("=", 1)[1].strip() for l in (ROOT / ".env").read_text().splitlines()
           if l.startswith("MICROBURBS_API_KEY="))


def get(path):
    url = BASE + urllib.parse.quote(path)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
                time.sleep(0.15)
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(float(e.headers.get("Retry-After", 2)))
                continue
            return {"data": None, "error": e.code}
        except Exception as e:
            print("  request failed:", repr(e)[:160])
            time.sleep(1 + attempt)
    return {"data": None, "error": "failed"}


def last_before(series, cutoff):
    pts = [p for p in (series or []) if p.get("date", "") <= cutoff and p.get("value")]
    return (pts[-1]["value"], pts[-1]["date"]) if pts else (None, None)


def main():
    suburbs = (get("/v1/me").get("area") or {}).get("suburbs") or SUBURBS
    out = {}
    for s in suburbs:
        ser = (get(f"/v1/suburbs/{s}/market/median-sale-price-series").get("data") or {})
        price, as_of = last_before(ser.get("unit"), CUTOFF)
        yl = (get(f"/v1/suburbs/{s}/market/yield-pct").get("data") or {})
        out[s] = {"unit_price": price, "price_date": as_of, "mb_unit_yield": yl.get("unit"),
                  "mb_yield_as_of": yl.get("as_of")}
        print(f"{s:28s} unit price {price} ({as_of}) | published unit yield {yl.get('unit')}")
    (ROOT / "cache").mkdir(exist_ok=True)
    (ROOT / "cache" / "suburb_prices.json").write_text(json.dumps(out, indent=1))
    print("WROTE cache/suburb_prices.json")


if __name__ == "__main__":
    main()
