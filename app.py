"""
Streamlit-dashboard: Formule 1 - de auto of de rijder? (casus 4, richting B)

Starten:   streamlit run app.py
Leest:     data/gekoppeld.csv  (uitvoer van het notebook, stap 4)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

st.set_page_config(page_title="F1: auto of rijder?", layout="wide")
DATA = Path(__file__).parent / "data" / "gekoppeld.csv"
ZEE, LAND, GRENS = "#101820", "#26323d", "#6b7c8a"


# ---------------------------------------------------------------- data + afgeleide kolommen
@st.cache_data
def laad():
    g = pd.read_csv(DATA)
    g["quali_diff"] = g.quali_pos - g.quali_pos_mate                    # negatief = rijder beter dan teamgenoot
    g["race_ok"] = g.race_gebruikbaar & g.race_gebruikbaar_mate
    g["race_diff"] = (g.positionOrder - g.positionOrder_mate).where(g.race_ok)
    return g


# ---------------------------------------------------------------- statistiek
@st.cache_data
def duel_toets(df, kol, min_duels, n_boot=1000):
    """Wint de betere rijder van een duo vaker van zijn teamgenoot dan toeval toelaat?"""
    rng = np.random.default_rng(42)
    d = df[(df.driverId < df.driverId_mate) & df[kol].notna()].copy()    # 1 rij per duel
    d["wint"] = d[kol] < 0
    g = (d.groupby(["constructorId", "year", "driverId", "driverId_mate"])
           .agg(n=("wint", "size"), w=("wint", "sum")).reset_index())
    g = g[g.n >= min_duels]
    if len(g) < 10:
        return None
    chi2 = (((g.w - g.n / 2) ** 2) / (g.n / 4)).sum()                   # bij gelijke rijders: elk duel een munt
    p = stats.chi2.sf(chi2, len(g))

    def verwacht(n):                                                    # E[max(W, n-W)] / n bij p = 0,5
        k = np.arange(n + 1)
        return (stats.binom.pmf(k, n, .5) * np.maximum(k, n - k)).sum() / n
    exp_n = {n: verwacht(n) for n in g.n.unique()}

    def extra(s):
        obs = np.maximum(s.w, s.n - s.w).sum() / s.n.sum()
        exp = (s.n.map(exp_n) * s.n).sum() / s.n.sum()
        return obs, exp
    obs, exp = extra(g)
    boot = [np.subtract(*extra(g.iloc[rng.integers(0, len(g), len(g))])) for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(duo_seizoenen=len(g), duels=int(g.n.sum()), chi2=chi2, p=p,
                betere_wint=obs, toeval=exp, extra=obs - exp, lo=lo, hi=hi)


@st.cache_data
def auto_aandeel(df, pos_kol, masker, n_boot=1000):
    """Aandeel van de spreiding in uitslagen dat tussen team-seizoenen (= de auto) zit."""
    rng = np.random.default_rng(42)
    q = df[masker].copy()
    if len(q) < 50:
        return None
    q["pct"] = q.groupby("raceId")[pos_kol].rank(pct=True)               # vergelijkbaar tussen jaren
    q["ts"] = q.constructorId.astype(str) + "_" + q.year.astype(str)
    s = q.groupby("ts").pct.agg(n="size", som="sum")
    s["ss"] = q.assign(k=q.pct ** 2).groupby("ts").k.sum()
    n, som, ss = s.n.values, s.som.values, s.ss.values

    def aandeel(c):
        N = (c * n).sum(); gem = (c * som).sum() / N
        return (c * n * (som / n - gem) ** 2).sum() / N / ((c * ss).sum() / N - gem ** 2)

    punt = aandeel(np.ones(len(n)))
    boot = [aandeel(rng.multinomial(len(n), np.ones(len(n)) / len(n))) for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(aandeel=punt, lo=lo, hi=hi)


# ---------------------------------------------------------------- kaart
@st.cache_data
def kaart_data(df, min_duels):
    ok = df[df.race_ok].copy()
    ok["pct"] = ok.groupby("raceId").positionOrder.rank(pct=True)
    pm = ok.set_index(["raceId", "driverId"]).pct.rename("pct_mate")
    d = ok[ok.driverId < ok.driverId_mate].merge(
        pm, left_on=["raceId", "driverId_mate"], right_index=True, how="left", validate="1:1")
    d["gat"] = (d.pct - d.pct_mate).abs()
    d["gat_corr"] = d.gat - d.groupby("year").gat.transform("mean")      # corrigeer voor tijdperk
    k = (d.groupby(["circuitId", "circuit", "land", "lat", "lng"])
           .agg(duels=("gat", "size"), gat_corr=("gat_corr", "mean"), jaar_gem=("year", "mean")).reset_index())
    k = k[k.duels >= min_duels].copy()
    k["verschil_pp"] = (k.gat_corr * 100).round(1)
    k["jaar_gem"] = k.jaar_gem.round(0).astype(int)
    return k


def maak_kaart(data, lim, europa=False):
    d = data.copy()
    if europa:
        d = d[d.lng.between(-12, 45) & d.lat.between(35, 65)]
    d["label"] = d.circuit.where(d.duels >= 60, "") if europa else ""
    fig = px.scatter_geo(
        d, lat="lat", lon="lng", size="duels", color="verschil_pp", text="label", hover_name="circuit",
        hover_data={"land": True, "duels": True, "jaar_gem": True, "verschil_pp": ":.1f",
                    "lat": False, "lng": False, "label": False},
        color_continuous_scale="RdBu_r", range_color=(-lim, lim), size_max=34 if europa else 24,
        labels={"verschil_pp": "Verschil (pp)", "duels": "Aantal duels", "jaar_gem": "Gem. jaar"})
    fig.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.92),
                      textposition="top center", textfont=dict(size=10, color="#e8edf2"))
    fig.update_geos(
        projection_type="mercator" if europa else "natural earth",
        showland=True, landcolor=LAND, showocean=True, oceancolor=ZEE,
        showcountries=True, countrycolor=GRENS, countrywidth=0.6,
        showcoastlines=True, coastlinecolor=GRENS, coastlinewidth=0.6,
        showlakes=False, showframe=False, bgcolor=ZEE,
        **(dict(lonaxis_range=[-11, 32], lataxis_range=[36, 59]) if europa else {}))
    fig.update_layout(paper_bgcolor=ZEE, font=dict(color="#e8edf2"), height=620 if europa else 520,
                      margin=dict(l=0, r=0, t=10, b=0),
                      coloraxis_colorbar=dict(title="Verschil<br>(procentpunt)", thickness=14, len=0.7))
    return fig


# ================================================================= pagina
if not DATA.exists():
    st.error(f"{DATA} niet gevonden. Draai eerst het notebook (stap 4) en sla `gekoppeld.csv` op in de map data/.")
    st.stop()

alle = laad()

st.title("Formule 1: de auto of de rijder?")
st.caption("Teamgenoten rijden in dezelfde auto. Door hen te vergelijken houden we de auto constant "
           "en meten we wat de rijder toevoegt. Bron: Kaggle `jtrotman/formula-1-race-data`, versie 134.")

with st.sidebar:
    st.header("Filters")
    j0, j1 = st.slider("Periode", 1950, 2023, (1950, 2023))
    min_duels = st.slider("Minimaal aantal duels per duo-seizoen", 5, 20, 8,
                          help="Duo-seizoenen met minder duels zijn te onbetrouwbaar en vallen af.")
    st.markdown("**Let op:** kwalificatiegegevens bestaan pas vanaf 1994.")

df = alle[alle.year.between(j0, j1)]
tq = duel_toets(df, "quali_diff", min_duels)
tr = duel_toets(df, "race_diff", min_duels)
aq = auto_aandeel(df, "quali_pos", df.quali_pos.notna() & df.quali_pos_mate.notna())
ar = auto_aandeel(df, "positionOrder", df.race_ok)

# ---- hoofdconclusie (getallen komen live uit de berekening, dus ze volgen de filters)
st.subheader("Hoofdconclusie")
if tq and aq:
    st.success(
        f"**In dezelfde auto wint de sterkere teamgenoot {tq['extra']*100:.1f} procentpunt meer kwalificatieduels "
        f"dan toeval zou opleveren** (95%-betrouwbaarheidsinterval {tq['lo']*100:.1f} tot {tq['hi']*100:.1f}). "
        + (f"Bij races is dat {tr['extra']*100:.1f} procentpunt ({tr['lo']*100:.1f} tot {tr['hi']*100:.1f}). " if tr else "")
        + f"Toch zit {aq['aandeel']:.0%} van de spreiding in startposities (BI {aq['lo']:.0%} tot {aq['hi']:.0%}) "
          f"tussen auto's: de rijder maakt aantoonbaar verschil, maar de auto blijft de grootste factor.")
elif tr:
    st.success(f"**Race:** de sterkere teamgenoot wint {tr['extra']*100:.1f} procentpunt meer duels dan toeval "
               f"(BI {tr['lo']*100:.1f} tot {tr['hi']*100:.1f}). Kwalificatiegegevens ontbreken in deze periode.")
else:
    st.warning("Te weinig duels in deze selectie. Vergroot de periode of verlaag het minimum aantal duels.")

# ---- cijfers
c1, c2 = st.columns(2)
for kol, naam, t, a in [(c1, "Kwalificatie (vanaf 1994)", tq, aq), (c2, "Race", tr, ar)]:
    with kol:
        st.markdown(f"#### {naam}")
        if t is None:
            st.info("Te weinig duels in deze selectie.")
            continue
        m1, m2, m3 = st.columns(3)
        m1.metric("Betere rijder wint", f"{t['betere_wint']:.1%}", f"toeval: {t['toeval']:.1%}", delta_color="off")
        m2.metric("Extra t.o.v. toeval", f"{t['extra']*100:.1f} pp", f"BI {t['lo']*100:.1f} tot {t['hi']*100:.1f}", delta_color="off")
        m3.metric("Aandeel auto", f"{a['aandeel']:.0%}" if a else "-", f"BI {a['lo']:.0%} tot {a['hi']:.0%}" if a else "", delta_color="off")
        st.caption(f"{t['duo_seizoenen']} duo-seizoenen, {t['duels']:,} duels. Chi-kwadraat = {t['chi2']:.0f}, p = {t['p']:.1e}.".replace(",", "."))

with st.expander("Hoe is dit berekend?"):
    st.markdown("""
- **Duel:** twee teamgenoten in dezelfde race (zelfde `raceId` + `constructorId`). Bij de race tellen alleen duels waarin beide rijders
  gefinisht hebben of uitvielen door een ongeluk of fout; uitval door een technisch defect telt niet mee (keuze B).
- **Toets:** als beide rijders even goed waren, is elk duel een munt (p = 0,5). De chi-kwadraattoets kijkt of de uitslagen per duo-seizoen
  daar te veel van afwijken. De betere rijder wint vanzelf meer dan 50%, daarom vergelijken we met de verwachting bij toeval (niet met 50%).
- **Betrouwbaarheidsinterval:** bootstrap (1000x) over duo-seizoenen.
- **Aandeel auto:** deel van de variantie in positie (als fractie van het veld) dat tussen team-seizoenen zit. Dit is een bovengrens:
  het gemiddelde van een team-seizoen bevat ook de kwaliteit van de twee rijders.
""")

# ---- kaart
st.subheader("Waar maakt de rijder meer uit?")
st.caption("Kleur: verschil tussen teamgenoten in de race op dit circuit, ten opzichte van het gemiddelde van hetzelfde seizoen "
           "(gecorrigeerd voor tijdperk). Rood = rijder maakt hier meer uit, blauw = minder. Grootte = aantal duels.")
kd = kaart_data(df, 20)
if len(kd) < 3:
    st.info("Te weinig circuits met genoeg duels in deze periode.")
else:
    lim = kd.verschil_pp.abs().max()
    tab1, tab2 = st.tabs(["Wereld", "Europa"])
    tab1.plotly_chart(maak_kaart(kd, lim), use_container_width=True)
    tab2.plotly_chart(maak_kaart(kd, lim, europa=True), use_container_width=True)
    st.caption("Veel verschillen zijn klein ten opzichte van de ruis (typische standaardfout per circuit: ca. 2 procentpunt). "
               "Alleen de uitschieters zijn duidelijk; kleine bollen zijn onzeker.")

# ---- afgeleide kolom in het verhaal
st.subheader("Het rijder-effect per rijder")
st.caption("Afgeleide kolom `quali_diff` / `race_diff`: eigen positie min die van de teamgenoot (negatief = beter). "
           "Gemiddeld per rijder over alle seizoenen in de selectie, minimaal 30 duels.")
kolom = st.radio("Meten op", ["race_diff", "quali_diff"], horizontal=True,
                 format_func=lambda x: "Race" if x == "race_diff" else "Kwalificatie")
r = (df[df[kolom].notna()].groupby(["coureur"]).agg(duels=(kolom, "size"), gem=(kolom, "mean")).reset_index())
r = r[r.duels >= 30].sort_values("gem")
if len(r):
    b1, b2 = st.columns(2)
    b1.markdown("**Verslaat zijn teamgenoot het vaakst**")
    b1.dataframe(r.head(10).rename(columns={"gem": "gem. verschil (plaatsen)"}).round(2), hide_index=True, use_container_width=True)
    b2.markdown("**Verliest het vaakst van zijn teamgenoot**")
    b2.dataframe(r.tail(10).iloc[::-1].rename(columns={"gem": "gem. verschil (plaatsen)"}).round(2), hide_index=True, use_container_width=True)
    st.caption("Let op: dit hangt ook af tegen wie je rijdt. Een rijder naast een topcoureur scoort slechter zonder slecht te zijn.")

# ---- verantwoording
with st.expander("Data-verantwoording (korte versie)"):
    st.markdown("""
**Bron:** Kaggle `jtrotman/formula-1-race-data`, versie 134, gedownload op 29-10-2026 (handmatig). Tabellen: results, races, drivers, constructors, circuits, status, qualifying.
**Periode:** 1950-2023 (de dataset loopt door tot 2026; die jaren zijn weggelaten). Kwalificatiegegevens bestaan pas vanaf 1994.
**Opschonen:** alleen echte starters; gedeelde auto's (170 rijen) eruit; alleen teams met precies 2 coureurs per race (19.146 rijen = 9.573 duels).
**Koppelen:** results naar drivers, constructors en circuits (veel-op-een); teamgenoot via `raceId` + `constructorId`, zelf uitgesloten.
**Keuzes:** technische uitval telt niet mee voor de racevergelijking; de indeling technisch/rijder is een aanname.
**Beperkingen:** een botsing is niet altijd de fout van de rijder; het auto-aandeel is een bovengrens; duo-seizoenen met te weinig duels vallen af.
""")
