"""Fetch suburb-level Microburbs data into cache/ (stdlib only, run on your Mac).

Usage:  python3 src/fetch_api.py
Reads MICROBURBS_API_KEY from .env. Respects ~10 req/s and retries on 429.
All series are truncated to <= 2026-07-31 later, in the feature step.
"""
import json, os, time, urllib.request, urllib.parse, urllib.error, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
BASE = "https://api-ht.microburbs.com.au"
CUTOFF = "2026-07-31"

def load_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("MICROBURBS_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("MICROBURBS_API_KEY missing in .env")

KEY = load_key()

def get(path, params=None, tries=5):
    url = BASE + urllib.parse.quote(path) + ("?" + urllib.parse.urlencode(params) if params else "")
    for i in range(tries):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {KEY}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                time.sleep(0.12)
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(float(e.headers.get("Retry-After", 2)))
                continue
            return {"data": None, "error": e.code, "url": url}
        except Exception as e:
            time.sleep(1 + i)
    return {"data": None, "error": "failed", "url": url}

SUBURB_ENDPOINTS = [
    ("market/median-rent-series", None),
    ("market/vacancy-rate-series", None),
    ("market/rent-map", None),
    ("market/bedroom-stats", None),
    ("market/supply-pressure", None),
    ("market/strata-levies", None),
    ("development/building-controls", None),
    ("development/construction-activity", None),
    ("lifestyle/cbd", None),
    ("shapes/busy-roads", {"geojson": "true"}),
    ("shapes/noise-heatmap", {"geojson": "true"}),
]

def paged(path, params, label, page=500):
    out, offset, total = [], 0, None
    while True:
        r = get(path, {**params, "limit": page, "offset": offset})
        d = r.get("data") or {}
        rows = d.get("results") or []
        if total is None:
            total = (d.get("meta") or {}).get("total", 0)
        if not rows:
            break
        out.extend(rows)
        offset += len(rows)
        if offset >= total:
            break
    print(f"  {label}: {len(out)}/{total}")
    return out

def main():
    me = get("/v1/me")
    names = ((me.get("area") or {}).get("suburbs")) or []
    print("team:", me.get("team"), "| suburbs:", len(names))
    CACHE.mkdir(exist_ok=True)
    for s in names:
        print(s)
        sd = CACHE / "suburbs" / s.replace("/", "_")
        sd.mkdir(parents=True, exist_ok=True)
        f = sd / "properties.json"
        if not f.exists():
            f.write_text(json.dumps(paged(f"/v1/suburbs/{s}/properties", {}, "properties")))
        f = sd / "rent_history.json"
        if not f.exists():
            f.write_text(json.dumps(paged(f"/v1/suburbs/{s}/listings/rent-history",
                                          {"to_date": CUTOFF}, "past adverts")))
        for ep, params in SUBURB_ENDPOINTS:
            f = sd / (ep.replace("/", "__") + ".json")
            if not f.exists():
                f.write_text(json.dumps(get(f"/v1/suburbs/{s}/{ep}", params)))
    print("done ->", CACHE)

if __name__ == "__main__":
    main()
