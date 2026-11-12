from __future__ import annotations

import heapq
import json
import time

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, riga, titolo

K = 20                # dimensione del seed set
MONTE_CARLO = 200     # simulazioni per stima dello spread
CANDIDATI = 800       # nodi valutabili (per grado)
N_RR = 40000          # insiemi reverse-reachable per RIS
ALPHA_SMOOTH = 5.0
SEED = 42


# ---------------------------------------------------------------------------
# grafo in formato CSR
# ---------------------------------------------------------------------------

def costruisci_csr(archi: pd.DataFrame):
    nodi = pd.unique(pd.concat([archi["u"], archi["v"]]))
    idx = {n: i for i, n in enumerate(nodi)}
    da = np.concatenate([archi["u"].map(idx).to_numpy(),
                         archi["v"].map(idx).to_numpy()]).astype(np.int32)
    a = np.concatenate([archi["v"].map(idx).to_numpy(),
                        archi["u"].map(idx).to_numpy()]).astype(np.int32)
    ordine = np.argsort(da, kind="stable")
    da, a = da[ordine], a[ordine]
    inizio = np.searchsorted(da, np.arange(len(nodi) + 1)).astype(np.int64)
    return nodi, idx, a, inizio, ordine


def probabilita(schema, nodi, idx, vicini, inizio, ordine, archi) -> np.ndarray:
    """Un valore per ogni arco diretto della struttura CSR."""
    n = len(nodi)
    grado = np.diff(inizio)

    if schema == "uniforme":
        return np.full(len(vicini), 0.01)

    if schema == "weighted_cascade":
        # p_uv = 1 / grado del RICEVENTE
        return 1.0 / np.maximum(grado[vicini], 1)

    if schema == "stimata":
        nodi_f = pd.read_parquet(DATA / "node_features.parquet").set_index("utente")

        delta = nodi_f["delta_vinti"].to_numpy()
        occasioni = np.maximum(nodi_f["n_thread"].to_numpy(), 1)
        base = float(delta.sum() / occasioni.sum())

        stima = np.full(n, base)
        posizioni = np.array([idx.get(u, -1) for u in nodi_f.index])
        validi = posizioni >= 0
        stima[posizioni[validi]] = (
            (delta[validi] + ALPHA_SMOOTH * base) /
            (occasioni[validi] + ALPHA_SMOOTH)
        )

        sorgente = np.repeat(np.arange(n), grado)
        p = stima[sorgente]

        return np.clip(p * (0.01 / p.mean()), 1e-4, 0.9)

    raise ValueError(schema)


# ---------------------------------------------------------------------------
# modelli di diffusione
# ---------------------------------------------------------------------------

def campiona_mondi(vicini, inizio, p, n, r, rng, modello):
    """Genera r "mondi possibili" (live-edge graphs), fissati una volta sola.

    PERCHE' SERVE. Simulare estraendo numeri casuali durante la propagazione
    sembra equivalente, ma non lo e' per un algoritmo goloso: il numero di
    estrazioni dipende da quanti archi si attraversano, quindi aggiungendo un
    seme l'intero flusso casuale si sposta e due candidati vengono valutati su
    mondi diversi. Il greedy finisce per scegliere il candidato con
    l'ESTRAZIONE piu' fortunata invece del guadagno piu' alto — la maledizione
    del vincitore — e dopo venti passi il seed set e' peggiore di una banale
    euristica per grado. E' esattamente il difetto che ho osservato.

    Fissando i mondi, lo spread diventa una funzione DETERMINISTICA dei semi,
    monotona e submodulare: greedy e CELF coincidono per costruzione e il
    confronto fra candidati e' esatto.

    IC: ogni arco e' vivo con probabilita' p_uv, indipendentemente.
    LT: ogni nodo sceglie AL PIU' un arco entrante, con probabilita' pari al
        peso normalizzato — e' la formulazione live-edge equivalente al modello
        a soglie lineari.
    """
    sorgenti = np.repeat(np.arange(n, dtype=np.int32), np.diff(inizio))

    if modello == "LT":
        # struttura calcolata UNA volta, non a ogni mondo
        ordine_in = np.argsort(vicini, kind="stable")
        in_p = p[ordine_in]
        b_ord = vicini[ordine_in]
        in_inizio = np.searchsorted(b_ord, np.arange(n + 1)).astype(np.int64)
        cum = np.cumsum(in_p)
        base = np.where(in_inizio[:-1] > 0, cum[np.maximum(in_inizio[:-1] - 1, 0)], 0.0)
        base[in_inizio[:-1] == 0] = 0.0
        ha_entranti = in_inizio[1:] > in_inizio[:-1]

    mondi = []
    for _ in range(r):
        if modello == "IC":
            vivo = rng.random(len(vicini)) < p
        else:
            u = rng.random(n)
            bersaglio = base + u
            pos = np.searchsorted(cum, bersaglio)
            ok = ha_entranti & (pos < in_inizio[1:])
            vivo_in = np.zeros(len(vicini), dtype=bool)
            vivo_in[pos[ok]] = True
            vivo = np.zeros(len(vicini), dtype=bool)
            vivo[ordine_in[vivo_in]] = True

        v_r = vicini[vivo]
        sorg = sorgenti[vivo]
        ordine = np.argsort(sorg, kind="stable")
        v_r = v_r[ordine]
        i_r = np.searchsorted(sorg[ordine], np.arange(n + 1)).astype(np.int64)
        mondi.append((v_r, i_r))
    return mondi


def raggiungibili(semi, v_r, i_r, n) -> int:
    """Quanti nodi sono raggiungibili dai semi in un mondo fissato."""
    visto = np.zeros(n, dtype=bool)
    visto[semi] = True
    frontiera = np.asarray(semi, dtype=np.int64)
    totale = len(semi)
    while len(frontiera):
        i0, i1 = i_r[frontiera], i_r[frontiera + 1]
        if (i1 - i0).sum() == 0:
            break
        pos = np.concatenate([np.arange(a, b) for a, b in zip(i0, i1)])
        nuovi = np.unique(v_r[pos])
        nuovi = nuovi[~visto[nuovi]]
        if not len(nuovi):
            break
        visto[nuovi] = True
        totale += len(nuovi)
        frontiera = nuovi
    return totale


def spread_mondi(semi, mondi, n) -> float:
    return float(np.mean([raggiungibili(semi, v, i, n) for v, i in mondi]))


def simula_ic(semi, vicini, inizio, p, n, rng) -> int:
    """Independent Cascade: ogni nodo appena attivato ha UNA sola occasione
    di attivare ciascun vicino, con probabilita' p sull'arco."""
    attivo = np.zeros(n, dtype=bool)
    attivo[semi] = True
    frontiera = np.array(semi, dtype=np.int64)
    totale = len(semi)

    while len(frontiera):
        i0, i1 = inizio[frontiera], inizio[frontiera + 1]
        conta = i1 - i0
        if conta.sum() == 0:
            break
        posizioni = np.concatenate([np.arange(a, b) for a, b in zip(i0, i1)])
        bersagli = vicini[posizioni]
        riesce = rng.random(len(posizioni)) < p[posizioni]
        nuovi = bersagli[riesce]
        nuovi = nuovi[~attivo[nuovi]]
        if len(nuovi) == 0:
            break
        nuovi = np.unique(nuovi)
        attivo[nuovi] = True
        totale += len(nuovi)
        frontiera = nuovi
    return totale


def simula_lt(semi, vicini, inizio, p, n, rng) -> int:
    """Linear Threshold: ogni nodo estrae una soglia e si attiva quando la
    somma dei pesi dei vicini attivi la supera."""
    soglia = rng.random(n)
    attivo = np.zeros(n, dtype=bool)
    attivo[semi] = True
    peso_acc = np.zeros(n)
    frontiera = np.array(semi, dtype=np.int64)
    totale = len(semi)

    while len(frontiera):
        i0, i1 = inizio[frontiera], inizio[frontiera + 1]
        if (i1 - i0).sum() == 0:
            break
        posizioni = np.concatenate([np.arange(a, b) for a, b in zip(i0, i1)])
        np.add.at(peso_acc, vicini[posizioni], p[posizioni])
        candidati = np.unique(vicini[posizioni])
        candidati = candidati[~attivo[candidati]]
        nuovi = candidati[peso_acc[candidati] >= soglia[candidati]]
        if len(nuovi) == 0:
            break
        attivo[nuovi] = True
        totale += len(nuovi)
        frontiera = nuovi
    return totale


def spread(semi, vicini, inizio, p, n, rng, modello, r=MONTE_CARLO,
           seme_comune: int | None = None) -> float:
    """Stima dello spread per simulazione.

    NUMERI CASUALI COMUNI. Con `seme_comune`, tutte le valutazioni che
    ricevono lo stesso valore usano ESATTAMENTE le stesse estrazioni.

    Serve perche' il greedy non deve stimare bene lo spread in assoluto: deve
    ORDINARE due candidati. Con estrazioni indipendenti, la differenza fra due
    candidati (2-3 nodi quando p=0,01) finisce sotto il rumore della stima e
    la selezione diventa casuale — al punto che CELF finiva sotto una banale
    euristica per grado, cosa impossibile se la selezione funziona.
    Accoppiando le estrazioni, il rumore comune si cancella nella differenza.
    """
    f = simula_ic if modello == "IC" else simula_lt
    gen = np.random.default_rng(seme_comune) if seme_comune is not None else rng
    return float(np.mean([f(semi, vicini, inizio, p, n, gen) for _ in range(r)]))


# ---------------------------------------------------------------------------
# algoritmi
# ---------------------------------------------------------------------------

def greedy(cand, mondi, n, k=K):
    """Greedy classico sui mondi fissati: a ogni passo rivaluta TUTTI i
    candidati. Con lo spread reso deterministico, la funzione obiettivo e'
    monotona e submodulare, quindi vale la garanzia (1 - 1/e)."""
    semi, valutazioni = [], 0
    for _ in range(k):
        base = spread_mondi(semi, mondi, n) if semi else 0.0
        migliore, guadagno = None, -1.0
        for c in cand:
            if c in semi:
                continue
            g = spread_mondi(semi + [c], mondi, n) - base
            valutazioni += 1
            if g > guadagno:
                migliore, guadagno = c, g
        semi.append(migliore)
    return semi, valutazioni


def celf(cand, mondi, n, k=K):
    """CELF: stesso risultato di greedy, molte meno valutazioni.

    Il guadagno marginale di un nodo non puo' crescere aggiungendo altri semi
    (submodularita'), quindi un guadagno calcolato a un passo precedente e' un
    LIMITE SUPERIORE di quello attuale: se e' gia' sotto il migliore corrente,
    quel nodo non puo' vincere e non va ricalcolato.

    La garanzia vale solo perche' lo spread e' deterministico sui mondi
    fissati. Con stime rumorose la submodularita' si rompe negli errori e
    CELF puo' divergere da greedy.
    """
    valutazioni = 0
    coda = []
    for c in cand:
        g = spread_mondi([c], mondi, n)
        valutazioni += 1
        heapq.heappush(coda, (-g, c, 0))          # -g: heapq e' un min-heap

    neg, primo, _ = heapq.heappop(coda)
    semi = [primo]
    corrente = -neg

    while len(semi) < k and coda:
        while True:
            neg_g, c, quando = coda[0]
            if quando == len(semi):               # aggiornato in questo passo
                heapq.heappop(coda)
                semi.append(c)
                corrente += -neg_g
                break
            heapq.heappop(coda)
            g = spread_mondi(semi + [c], mondi, n) - corrente
            valutazioni += 1
            heapq.heappush(coda, (-g, c, len(semi)))
    return semi, valutazioni


def csr_entrante(vicini, inizio, p, n):
    """CSR degli archi ENTRANTI, con la probabilita' dell'arco u->v.

    Serve a RIS: un insieme reverse-reachable si costruisce risalendo da un
    bersaglio verso chi potrebbe averlo attivato, quindi si percorrono archi
    all'indietro con la probabilita' dell'arco NELLA SUA DIREZIONE ORIGINALE.
    Con weighted cascade p_uv = 1/grado(v) e' asimmetrica, quindi usare per
    sbaglio le probabilita' degli archi uscenti falsa tutto: e' l'errore che
    faceva produrre a RIS semi da spread 75 contro i 2741 di CELF.
    """
    sorgenti = np.repeat(np.arange(n, dtype=np.int32), np.diff(inizio))
    bersagli = vicini
    ordine = np.argsort(bersagli, kind="stable")
    b_ord = bersagli[ordine]
    return (sorgenti[ordine].astype(np.int32),
            p[ordine],
            np.searchsorted(b_ord, np.arange(n + 1)).astype(np.int64))


def ris(vicini, inizio, p, n, rng, k=K, n_rr=N_RR):
    """Reverse Influence Sampling — il nucleo di TIM/TIM+/IMM.

    Si campionano bersagli a caso e si risale a chi POTREBBE averli attivati.
    Un nodo che compare in molti insiemi reverse-reachable influenza molti
    bersagli: il problema diventa max-coverage, risolvibile in modo goloso
    senza simulare nemmeno una cascata in avanti.
    """
    in_vic, in_p, in_inizio = csr_entrante(vicini, inizio, p, n)

    copertura = [[] for _ in range(n)]
    for r in range(n_rr):
        sorgente = int(rng.integers(0, n))
        visti = {sorgente}
        frontiera = [sorgente]
        while frontiera:
            nuovi = []
            for x in frontiera:
                i0, i1 = in_inizio[x], in_inizio[x + 1]
                if i1 <= i0:
                    continue
                cand = in_vic[i0:i1]
                colpiti = cand[rng.random(i1 - i0) < in_p[i0:i1]]
                for y in colpiti:
                    y = int(y)
                    if y not in visti:
                        visti.add(y)
                        nuovi.append(y)
            frontiera = nuovi
        for x in visti:
            copertura[x].append(r)

    coperti = np.zeros(n_rr, dtype=bool)
    coda = [(-len(c), i, 0) for i, c in enumerate(copertura) if c]
    heapq.heapify(coda)
    semi = []

    while len(semi) < k and coda:
        neg, i, quando = heapq.heappop(coda)
        if quando == len(semi):
            semi.append(i)
            for r in copertura[i]:
                coperti[r] = True
            continue
        vero = sum(1 for r in copertura[i] if not coperti[r])
        if vero:
            heapq.heappush(coda, (-vero, i, len(semi)))
    return semi


def spread_con_errore(semi, vicini, inizio, p, n, modello, r=600, seme=SEED + 7):
    """Media ED errore standard. Senza l'errore non si distingue un vantaggio
    reale da rumore: e' esattamente l'equivoco in cui sono caduto la prima
    volta, quando CELF sembrava perdere contro una banale euristica."""
    f = simula_ic if modello == "IC" else simula_lt
    gen = np.random.default_rng(seme)
    v = np.array([f(semi, vicini, inizio, p, n, gen) for _ in range(r)], dtype=float)
    return float(v.mean()), float(v.std() / np.sqrt(r))


def jaccard(a, b) -> float:
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


# ---------------------------------------------------------------------------

def main() -> None:
    titolo("STEP 7 — INFLUENCE MAXIMIZATION")

    archi = pd.read_parquet(DATA / "grafo_archi.parquet")
    nodi, idx, vicini, inizio, ordine = costruisci_csr(archi)
    n = len(nodi)
    grado = np.diff(inizio)
    cand = list(np.argsort(grado)[::-1][:CANDIDATI])
    semi_deg = list(np.argsort(grado)[::-1][:K])

    riga("nodi", n)
    riga("archi diretti (CSR)", len(vicini))
    riga("candidati (per grado)", len(cand))
    riga("k", K)
    riga("mondi campionati", MONTE_CARLO)
    print()
    print("  Selezione e valutazione usano mondi DIVERSI: ottimizzare e")
    print("  misurare sullo stesso campione gonfierebbe il risultato.")

    risultati, seed_per_schema = {}, {}

    for schema in ("uniforme", "weighted_cascade", "stimata"):
        p = probabilita(schema, nodi, idx, vicini, inizio, ordine, archi)
        titolo(f"ASSEGNAZIONE: {schema}")
        riga("p media", f"{p.mean():.4f}")
        riga("p mediana", f"{np.median(p):.4f}")

        for modello in ("IC", "LT"):
            t0 = time.time()
            mondi_fit = campiona_mondi(vicini, inizio, p, n, MONTE_CARLO,
                                       np.random.default_rng(SEED), modello)
            mondi_val = campiona_mondi(vicini, inizio, p, n, MONTE_CARLO,
                                       np.random.default_rng(SEED + 999), modello)
            t_mondi = time.time() - t0

            t0 = time.time()
            semi, val = celf(cand, mondi_fit, n)
            t_celf = time.time() - t0

            sp = spread_mondi(semi, mondi_val, n)
            sp_deg = spread_mondi(semi_deg, mondi_val, n)
            sp_in = spread_mondi(semi, mondi_fit, n)
            deg_in = spread_mondi(semi_deg, mondi_fit, n)

            chiave = f"{schema}|{modello}"
            risultati[chiave] = {
                "seed_celf": [str(nodi[i]) for i in semi],
                "spread_celf": sp, "spread_degree": sp_deg,
                "spread_celf_dentro": sp_in, "spread_degree_dentro": deg_in,
                "valutazioni_celf": val, "secondi_celf": t_celf,
            }
            seed_per_schema[chiave] = semi

            print()
            print(f"  {modello}:")
            riga("  spread CELF", f"{sp:.1f}", f"nodi attivati (k={K})")
            riga("  spread top-degree", f"{sp_deg:.1f}")
            riga("  vantaggio di CELF", f"{sp - sp_deg:+.1f}",
                 f"({sp / max(sp_deg, 1e-9) - 1:+.1%})")
            riga("  (controllo: dentro campione)",
                 f"CELF {sp_in:.1f} vs degree {deg_in:.1f}",
                 "OK" if sp_in >= deg_in else "ANOMALO")
            riga("  valutazioni", val, f"vs {len(cand) * K:,} del greedy")
            riga("  risparmio di CELF", f"{1 - val / (len(cand) * K):.1%}")
            riga("  tempo", f"{t_celf:.0f}s", f"(+{t_mondi:.0f}s per i mondi)")

    # ------------------------------------------------------------------
    titolo("GREEDY vs CELF — devono coincidere")
    print("  Sui mondi fissati lo spread e' deterministico e submodulare,")
    print("  quindi CELF non e' un'approssimazione di greedy: e' lo stesso")
    print("  risultato con meno valutazioni. Se differiscono, c'e' un errore.")
    print()
    p = probabilita("weighted_cascade", nodi, idx, vicini, inizio, ordine, archi)
    mondi = campiona_mondi(vicini, inizio, p, n, 60,
                           np.random.default_rng(SEED), "IC")
    piccolo = cand[:150]
    t0 = time.time(); g_semi, g_val = greedy(piccolo, mondi, n, k=5); t_g = time.time() - t0
    t0 = time.time(); c_semi, c_val = celf(piccolo, mondi, n, k=5); t_c = time.time() - t0
    riga("seed greedy", str([str(nodi[i]) for i in g_semi]))
    riga("seed CELF", str([str(nodi[i]) for i in c_semi]))
    riga("valutazioni greedy", g_val, f"in {t_g:.0f}s")
    riga("valutazioni CELF", c_val, f"in {t_c:.0f}s")
    riga("stessi semi", "SI" if g_semi == c_semi else "NO — errore")
    riga("valutazioni risparmiate", f"{1 - c_val / g_val:.1%}")
    risultati["greedy_vs_celf"] = {
        "greedy": [str(nodi[i]) for i in g_semi], "celf": [str(nodi[i]) for i in c_semi],
        "val_greedy": g_val, "val_celf": c_val, "identici": g_semi == c_semi,
    }

    # ------------------------------------------------------------------
    titolo("RIS — il nucleo di TIM/IMM")
    for schema in ("uniforme", "stimata"):
        p = probabilita(schema, nodi, idx, vicini, inizio, ordine, archi)
        mondi_val = campiona_mondi(vicini, inizio, p, n, MONTE_CARLO,
                                   np.random.default_rng(SEED + 999), "IC")
        t0 = time.time()
        semi_ris = ris(vicini, inizio, p, n, np.random.default_rng(SEED))
        t_ris = time.time() - t0
        sp_ris = spread_mondi(semi_ris, mondi_val, n)
        sp_celf = risultati[f"{schema}|IC"]["spread_celf"]
        print()
        print(f"  {schema}:")
        riga("  spread RIS", f"{sp_ris:.1f}", f"in {t_ris:.0f}s ({N_RR:,} insiemi RR)")
        riga("  spread CELF", f"{sp_celf:.1f}",
             f"in {risultati[f'{schema}|IC']['secondi_celf']:.0f}s")
        riga("  qualita' RIS / CELF", f"{sp_ris / max(sp_celf, 1e-9):.1%}")
        riga("  Jaccard con CELF",
             f"{jaccard(semi_ris, seed_per_schema[f'{schema}|IC']):.2f}")
        risultati[f"ris_{schema}"] = {"spread": sp_ris, "secondi": t_ris,
                                      "seed": [str(nodi[i]) for i in semi_ris]}

    # ------------------------------------------------------------------
    titolo("L'ASSEGNAZIONE DELLE PROBABILITA' E' INNOCUA?")
    print("  Jaccard fra i seed set prodotti da assegnazioni diverse.")
    print("  Vicino a 1 = la scelta non conta. Vicino a 0 = conta molto.")
    print()
    schemi = ("uniforme", "weighted_cascade", "stimata")
    for modello in ("IC", "LT"):
        print(f"  {modello}:")
        acc = {}
        for i, a in enumerate(schemi):
            for b in schemi[i + 1:]:
                j = jaccard(seed_per_schema[f"{a}|{modello}"],
                            seed_per_schema[f"{b}|{modello}"])
                print(f"    {a:<18} vs {b:<18} {j:>6.2f}")
                acc[f"{a}|{b}"] = j
        risultati[f"jaccard_{modello}"] = acc

    (DATA / "im_risultati.json").write_text(
        json.dumps(risultati, indent=2), encoding="utf-8")
    riga("im_risultati.json", "scritto")

    print()
    print("  RICORDA: manca la ground truth delle cascate. Questi numeri")
    print("  confrontano algoritmi e assegnazioni fra loro, non misurano")
    print("  diffusione reale su Reddit.")


if __name__ == "__main__":
    main()
