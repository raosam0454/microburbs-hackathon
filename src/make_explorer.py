"""Build explorer.html: a self-contained Rent Explorer (map, search, flat card,
parking ROI). Run after rents_model.py. Data embedded is our own estimates plus
Open G-NAF addresses and coordinates; no past adverts are embedded."""
import json, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def main():
    P = pd.read_pickle(OUT / "final_predictions.pkl")
    met = json.load(open(OUT / "metrics.json"))
    P = P.sort_values(["suburb", "building_address", "address"])
    wo = P["rent_without_car_space"].astype(int)
    lo = ((wo + P["lo_off"]).clip(lower=100) / 5).round() * 5
    hi = ((wo + P["hi_off"]) / 5).round() * 5
    src = P["source"].map({"own_flat": 2, "building": 1, "area": 0}).astype(int)
    car = P["car_space"].map(lambda v: -1 if pd.isna(v) else int(v))
    fl = P["floor_level"].map(lambda v: -1 if pd.isna(v) else int(v))
    sub_names = sorted(P["suburb"].unique())
    sub_idx = {s: i for i, s in enumerate(sub_names)}
    data = {
        "id": P["gnaf_id"].str.replace("GANSW", "", regex=False).tolist(),
        "a": P["address"].str.title().tolist(),
        "b": P["building_address"].fillna("").str.title().tolist(),
        "s": P["suburb"].map(sub_idx).tolist(),
        "y": P["latitude"].round(5).tolist(),
        "x": P["longitude"].round(5).tolist(),
        "bd": P["bedrooms"].fillna(1).astype(int).tolist(),
        "ba": P["bathrooms"].fillna(1).astype(int).tolist(),
        "c": car.tolist(),
        "f": fl.tolist(),
        "w": P["rent_with_car_space"].astype(int).tolist(),
        "o": wo.tolist(),
        "lo": lo.astype(int).tolist(),
        "hi": hi.astype(int).tolist(),
        "e": src.tolist(),
        "n": P["bld_n"].fillna(0).astype(int).tolist(),
    }
    m = met["model"]
    meta = {"suburbs": sub_names, "mae": round(m["median_abs_err"]), "w100": round(m["within_100"] * 100, 1),
            "r2": round(m["r2"], 3), "cov": round(met["range_coverage"] * 100, 1), "nval": m["n"]}
    yf = OUT / "suburb_yield.json"
    meta["yld"] = json.load(open(yf)) if yf.exists() else {}
    vend = ROOT / "src" / "vendor"
    html = TEMPLATE.replace("__LCSS__", (vend / "leaflet.css").read_text()).replace(
        "__LJS__", (vend / "leaflet.js").read_text())
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace(
        "__META__", json.dumps(meta))
    (ROOT / "explorer.html").write_text(html, encoding="utf-8")
    print("WROTE explorer.html", round(len(html) / 1e6, 1), "MB")


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rent Explorer, City of Sydney (Team HanSolo)</title>
<style>__LCSS__</style>
<script>__LJS__</script>
<style>
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1d2733}
header{background:#1f4e79;color:#fff;padding:10px 16px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
header h1{font-size:18px;margin:0}header .kpi{font-size:12px;opacity:.9}
#wrap{display:flex;height:calc(100vh - 52px)}#map{flex:1}
#side{width:400px;overflow:auto;border-left:1px solid #dde3ea;padding:14px}
#q{width:100%;padding:10px;font-size:15px;border:1px solid #b8c4d0;border-radius:8px}
#hits div{padding:7px 8px;border-bottom:1px solid #eef1f5;cursor:pointer;font-size:14px}#hits div:hover{background:#f3f6fa}
.card{background:#f3f6fa;border-radius:10px;padding:14px;margin-top:12px}
.big{font-size:30px;font-weight:700;color:#1f4e79}.muted{color:#5b6875;font-size:13px}
.row{display:flex;gap:10px}.row>div{flex:1}.pill{display:inline-block;background:#e3ebf4;border-radius:12px;padding:2px 9px;font-size:12px;margin:2px}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:5px;border-bottom:1px solid #e3e8ee;text-align:left}
tr.f{cursor:pointer}tr.f:hover{background:#eef3f8}.bar{height:8px;background:#dde3ea;border-radius:4px;position:relative;margin:8px 0}
.bar span{position:absolute;height:8px;background:#7fa6cc;border-radius:4px}.bar i{position:absolute;top:-4px;width:3px;height:16px;background:#1f4e79}
input.n{width:110px;padding:4px}
details.tips{margin-top:12px;background:#fff8e1;border:1px solid #f3e0a1;border-radius:10px;padding:8px 12px}
details.tips summary{cursor:pointer;font-weight:600;list-style:none;display:flex;align-items:center;gap:6px}
details.tips summary::-webkit-details-marker{display:none}
details.tips summary::after{content:"Show";margin-left:auto;font-weight:400;font-size:12px;color:#8a6d1d}
details.tips[open] summary::after{content:"Hide"}
details.tips ul{padding-left:18px;margin:8px 0;font-size:14px;line-height:1.5}
.bulb{font-size:18px}
header .kpi b{color:#ffd166}.legend{background:#fff;padding:6px 8px;border-radius:6px;font-size:12px;line-height:18px}
.legend b{display:inline-block;width:12px;height:12px;border-radius:6px;margin-right:6px;vertical-align:middle}
</style></head><body>
<header><div><h1>Sydney Flat Rents</h1><div class="kpi">What could this flat rent for? Search any of 65,777 flats in the City of Sydney.</div></div>
<div class="kpi" id="kpi" style="margin-left:auto;text-align:right"></div></header>
<div id="wrap"><div id="map"></div><div id="side">
<input id="q" placeholder="Search an address, e.g. 8 Sam Sing Street" autocomplete="off">
<div id="hits"></div>
<details class="tips"><summary><span class="bulb">💡</span> How to use</summary>
<ul><li><b>Search</b> a street or building, then pick a flat.</li>
<li><b>Click a dot</b> on the map to see every flat in that building.</li>
<li>Each flat shows its weekly rent <b>with and without parking</b>, a likely range, and why we think so.</li>
<li>Type a car space price to see what <b>parking earns</b> as an investment.</li></ul>
<p class="muted">Dot colour shows typical rent per bedroom. Estimates use adverts up to 31 July 2026, brought up to August 2026 prices.</p></details>
<div id="panel">
<div id="sub"></div></div></div></div>
<script>
const D=__DATA__, M=__META__, N=D.id.length;
document.getElementById('kpi').innerHTML=`Tested on real adverts: usually within <b>$${M.mae}</b> a week<br>${Math.round(M.w100)}% of estimates within $100`;
const map=L.map('map',{preferCanvas:true}).setView([-33.885,151.205],14);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors | Addresses: Open G-NAF, Geoscape Australia'}).addTo(map);
// buildings
const B=new Map();for(let i=0;i<N;i++){const k=D.b[i]+'|'+D.s[i];if(!B.has(k))B.set(k,[]);B.get(k).push(i)}
const med=a=>{const s=[...a].sort((x,y)=>x-y);return s[Math.floor(s.length/2)]};
const per=i=>D.o[i]/Math.max(1,D.bd[i]===0?0.75:D.bd[i]);
const vals=[...B.values()].map(ix=>med(ix.map(per))),q=[...vals].sort((a,b)=>a-b);
const cut=[.2,.4,.6,.8].map(p=>q[Math.floor(p*q.length)]),cols=['#2c7bb6','#abd9e9','#ffffbf','#fdae61','#d7191c'];
const col=v=>cols[cut.filter(c=>v>c).length];
let hl=null;
[...B.entries()].forEach(([k,ix],j)=>{const i=ix[0];L.circleMarker([D.y[i],D.x[i]],{radius:Math.min(12,3+Math.sqrt(ix.length)/2),color:'#333',weight:.5,fillColor:col(vals[j]),fillOpacity:.85})
 .addTo(map).bindTooltip(`${D.b[i]||D.a[i]}: ${ix.length} flats`).on('click',()=>showBuilding(ix))});
map.fitBounds(L.latLngBounds(D.y.map((y,i)=>[y,D.x[i]])),{padding:[20,20]});
const lg=L.control({position:'bottomleft'});lg.onAdd=()=>{const d=L.DomUtil.create('div','legend');d.innerHTML='<b style="background:none"></b>Typical weekly rent per bedroom<br>'+cols.map((c,i)=>`<b style="background:${c}"></b>${i==0?'under $'+Math.round(cut[0]):i==4?'over $'+Math.round(cut[3]):'$'+Math.round(cut[i-1])+' to $'+Math.round(cut[i])}`).join('<br>');return d};lg.addTo(map);
const money=v=>'$'+Math.round(v).toLocaleString();
const ev=i=>D.e[i]==2?'This flat\'s own past adverts, re-priced to Aug 2026':D.e[i]==1?`${D.n[i]} past adverts in this building`:'Nearby mesh block and suburb adverts (no history in this building)';
function showFlat(i){
 if(hl)map.removeLayer(hl);hl=L.circleMarker([D.y[i],D.x[i]],{radius:14,color:'#1f4e79',weight:3,fill:false}).addTo(map);map.setView([D.y[i],D.x[i]],17);
 const prem=D.w[i]-D.o[i],span=Math.max(1,D.hi[i]-D.lo[i]+200),base=D.lo[i]-100;
 const has=D.c[i]==1?'Has a car space':D.c[i]==0?'No car space':'Parking unknown';
 document.getElementById('panel').innerHTML=`<div class="card"><div class="muted">${M.suburbs[D.s[i]]}</div><b>${D.a[i]}</b><div>
 <span class="pill">${D.bd[i]==0?'Studio':D.bd[i]+' bed'}</span><span class="pill">${D.ba[i]} bath</span><span class="pill">${has}</span>${D.f[i]>=0?`<span class="pill">Level ${D.f[i]}</span>`:''}</div></div>
 <div class="card"><div class="row"><div><div class="muted">Without car space</div><div class="big">${money(D.o[i])}</div><div class="muted">per week</div></div>
 <div><div class="muted">With car space</div><div class="big">${money(D.w[i])}</div><div class="muted">per week</div></div></div>
 <div class="bar"><span style="left:${(D.lo[i]-base)/span*100}%;width:${(D.hi[i]-D.lo[i])/span*100}%"></span><i style="left:${(D.o[i]-base)/span*100}%"></i></div>
 <div class="muted">Likely range without parking: <b>${money(D.lo[i])} to ${money(D.hi[i])}</b> a week (80% range)</div>
 <p class="muted">Evidence: ${ev(i)}.</p></div>
 <div class="card"><b>Is a car space worth it here?</b><p>Parking adds <b>${money(prem)}</b> a week, or <b>${money(prem*52)}</b> a year.</p>
 Car space price: $<input class="n" id="cp" type="number" value="100000" step="5000"> <span id="roi"></span>
 <p class="muted">Gross yield on the space before strata and costs.${prem<15?' Most flats nearby already come with a space, so parking adds little here.':''}</p></div>
 <div class="card"><b>Annual rent</b><p>${money(D.o[i]*52)} without parking, ${money(D.w[i]*52)} with.</p>${(()=>{const sn=M.suburbs[D.s[i]].replace(/ \(.*\)/,''),y=M.yld[sn];return y?`<p>Typical gross yield for flats in ${sn}: <b>${y.toFixed(1)}%</b> a year (our rents vs median unit price).</p>`:''})()}</div>`;
 const r=()=>{const p=+document.getElementById('cp').value||1;document.getElementById('roi').innerHTML=`= <b>${(prem*52/p*100).toFixed(1)}%</b> gross yield`};
 document.getElementById('cp').oninput=r;r();
}
function showBuilding(ix){const i0=ix[0];
 const rows=ix.slice().sort((a,b)=>D.bd[a]-D.bd[b]||D.o[a]-D.o[b]).slice(0,300).map(i=>`<tr class="f" onclick="showFlat(${i})"><td>${D.a[i].split(' ').slice(0,3).join(' ')}</td><td>${D.bd[i]==0?'Studio':D.bd[i]}</td><td>${money(D.o[i])}</td><td>${money(D.w[i])}</td></tr>`).join('');
 const by={};ix.forEach(i=>{(by[D.bd[i]]=by[D.bd[i]]||[]).push(D.o[i])});
 document.getElementById('panel').innerHTML=`<div class="card"><div class="muted">${M.suburbs[D.s[i0]]}</div><b>${D.b[i0]||D.a[i0]}</b><p class="muted">${ix.length} flats. Median rent without parking: ${Object.entries(by).map(([b,v])=>`${b==0?'studio':b+' bed'} ${money(med(v))}`).join(', ')}.</p></div>
 <div class="card"><table><tr><th>Flat</th><th>Beds</th><th>No parking</th><th>With</th></tr>${rows}</table></div>`;
 map.setView([D.y[i0],D.x[i0]],17);
}
const qi=document.getElementById('q'),hits=document.getElementById('hits'),lower=D.a.map(a=>a.toLowerCase());
qi.oninput=()=>{const t=qi.value.toLowerCase().trim();if(t.length<3){hits.innerHTML='';return}
 const parts=t.split(/\s+/),r=[];for(let i=0;i<N&&r.length<12;i++){if(parts.every(p=>lower[i].includes(p)))r.push(i)}
 hits.innerHTML=r.map(i=>`<div onclick="showFlat(${i});hits.innerHTML=''">${D.a[i]}</div>`).join('')||'<div class="muted">No match</div>'};
</script></body></html>"""

if __name__ == "__main__":
    main()
