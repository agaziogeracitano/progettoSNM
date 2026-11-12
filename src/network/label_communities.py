from __future__ import annotations

import json
import time
from collections import defaultdict

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import numpy as np
import pandas as pd
import requests

from common import DATA, W_GRAFO, carica_utterance, chiave_groq, e_escluso, \
    modello_groq, nome_autore, riga, titolo

SEED = 42
DIM_MINIMA = 10          # comunita' piu' piccole non vengono etichettate
CAMPIONI_PER_COMUNITA = 12
LUNGHEZZA_MIN = 80        # scarta commenti troppo corti (poco informativi)
LUNGHEZZA_MAX = 600       # tronca quelli lunghi, per un prompt compatto
ALPHA_SMOOTH = 5.0
URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM = """You are given a sample of Reddit comments written by members of
the same community, detected automatically from a reply network on
r/ChangeMyView (a debate subreddit). Identify the dominant topic or theme
these comments share.

Reply with ONLY a JSON object:
{"etichetta": "2-4 word topic label in Italian", "descrizione": "one sentence in Italian describing the theme"}

No commentary outside the JSON."""


def costruisci_grafo() -> nx.Graph:
    archi = pd.read_parquet(DATA / "grafo_archi.parquet")
    G = nx.Graph()
    for u, v, w in archi.itertuples(index=False):
        G.add_edge(u, v, weight=float(w))
    return G


def raccogli_campioni(comunita_di: dict[str, int], target: set[int]) -> dict[int, list[str]]:
    """Un passaggio sul corpus (solo thread di W_grafo) per raccogliere
    qualche testo scritto dai membri di ciascuna comunita' abbastanza
    grande. Non serve rileggere tutto il corpus per il grafo — quello e'
    gia' in grafo_archi.parquet — ma per i TESTI non esiste una cache piu'
    piccola: units.parquet contiene solo le unita' etichettate (una
    minoranza dei nodi del grafo), la maggioranza degli utenti compare nel
    grafo senza mai comparire in una coppia."""
    threads = pd.read_parquet(DATA / "threads.parquet")
    grafo_roots = set(threads.loc[threads["split"] == W_GRAFO, "root"])

    campioni: dict[int, list[str]] = defaultdict(list)
    utt = carica_utterance()
    for u in utt.values():
        if u["root"] not in grafo_roots:
            continue
        autore = nome_autore(u)
        if e_escluso(autore) or autore not in comunita_di:
            continue
        cid = comunita_di[autore]
        if cid not in target or len(campioni[cid]) >= CAMPIONI_PER_COMUNITA:
            continue
        testo = " ".join((u.get("text") or "").split())
        if len(testo) < LUNGHEZZA_MIN:
            continue
        campioni[cid].append(testo[:LUNGHEZZA_MAX])
    return campioni


def etichetta_comunita(sessione, chiave, modello, testi: list[str]) -> dict:
    numerati = "\n\n".join(f"{i + 1}. {t}" for i, t in enumerate(testi))
    for tentativo in range(3):
        r = sessione.post(
            URL, headers={"Authorization": f"Bearer {chiave}"},
            json={"model": modello, "temperature": 0.2, "max_tokens": 200,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": numerati}]},
            timeout=60,
        )
        if r.status_code == 429:
            time.sleep(float(r.headers.get("retry-after", 15)) + 1)
            continue
        if not r.ok:
            time.sleep(3)
            continue
        try:
            d = json.loads(r.json()["choices"][0]["message"]["content"])
            if "etichetta" in d and "descrizione" in d:
                return d
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
        time.sleep(2)
    return {"etichetta": "(non classificata)", "descrizione": ""}


def main() -> None:
    titolo("ETICHETTATURA TEMATICA DELLE COMUNITA'")

    chiave, modello = chiave_groq(), modello_groq()

    print("\ncostruzione del grafo e rilevamento comunita'...")
    G = costruisci_grafo()
    comunita = nx.community.louvain_communities(G, weight="weight", seed=SEED)
    comunita_di = {n: i for i, c in enumerate(comunita) for n in c}

    grandi = {i for i, c in enumerate(comunita) if len(c) >= DIM_MINIMA}
    riga("comunita' totali", len(comunita))
    riga("comunita' da etichettare (>=%d nodi)" % DIM_MINIMA, len(grandi))

    # ------------------------------------------------------------------
    titolo("TASSO DI PERSUASIONE PER COMUNITA'")
    print("  Stesso stimatore, corretto, gia' usato in influence/maximize.py:")
    print("  denominatore = discussioni frequentate, non unita' appaiate.\n")

    nodi_f = pd.read_parquet(DATA / "node_features.parquet").set_index("utente")
    base = float(nodi_f["delta_vinti"].sum() / nodi_f["n_thread"].clip(lower=1).sum())
    p_hat = (nodi_f["delta_vinti"] + ALPHA_SMOOTH * base) / \
            (nodi_f["n_thread"].clip(lower=1) + ALPHA_SMOOTH)
    riga("tasso medio globale (smussato)", f"{base:.2%}")

    tasso_comunita: dict[int, float] = {}
    dim_comunita: dict[int, int] = {}
    for i, membri in enumerate(comunita):
        dim_comunita[i] = len(membri)
        presenti = [m for m in membri if m in p_hat.index]
        if presenti:
            tasso_comunita[i] = float(p_hat.loc[presenti].mean())

    # ------------------------------------------------------------------
    titolo("CAMPIONAMENTO DEI TESTI (un passaggio sul corpus)")
    campioni = raccogli_campioni(comunita_di, grandi)
    for cid in sorted(grandi):
        riga(f"  comunita' {cid}", len(campioni.get(cid, [])), "testi raccolti")

    # ------------------------------------------------------------------
    titolo("ETICHETTATURA VIA LLM")
    riga("modello", modello)
    sessione = requests.Session()
    risultati = {}

    for k, cid in enumerate(sorted(grandi, key=lambda c: -dim_comunita[c]), 1):
        testi = campioni.get(cid, [])
        if len(testi) < 3:
            print(f"  [{k}/{len(grandi)}] comunita' {cid}: troppo pochi testi, saltata")
            continue
        etichetta = etichetta_comunita(sessione, chiave, modello, testi)
        risultati[cid] = {
            "dimensione": dim_comunita[cid],
            "n_campioni": len(testi),
            "tasso_persuasione": tasso_comunita.get(cid),
            "etichetta": etichetta["etichetta"],
            "descrizione": etichetta["descrizione"],
        }
        print(f"  [{k}/{len(grandi)}] comunita' {cid} "
              f"({dim_comunita[cid]} nodi): {etichetta['etichetta']}")
        time.sleep(0.5)   # cortesia verso l'API, non necessario per il budget

    # ------------------------------------------------------------------
    titolo("SCRITTURA")

    mappa = pd.DataFrame(
        [(u, c) for u, c in comunita_di.items()],
        columns=["utente", "comunita_id"],
    )
    mappa.to_parquet(DATA / "comunita.parquet", index=False)
    riga("comunita.parquet", len(mappa), "righe (nodo -> comunita')")

    (DATA / "comunita_etichette.json").write_text(
        json.dumps(risultati, indent=2, ensure_ascii=False), encoding="utf-8")
    riga("comunita_etichette.json", len(risultati), "comunita' etichettate")

    # ------------------------------------------------------------------
    titolo("RIEPILOGO — comunita' per dimensione")
    print(f"  {'id':<5}{'nodi':>7}{'tasso':>9}   {'tema':<28}descrizione")
    print("  " + "-" * 100)
    for cid, r in sorted(risultati.items(), key=lambda x: -x[1]["dimensione"]):
        tasso = f"{r['tasso_persuasione']:.1%}" if r["tasso_persuasione"] else "  n/d"
        print(f"  {cid:<5}{r['dimensione']:>7}{tasso:>9}   "
              f"{r['etichetta']:<28}{r['descrizione'][:60]}")

    tassi = [r["tasso_persuasione"] for r in risultati.values() if r["tasso_persuasione"]]
    if len(tassi) >= 5:
        print(f"\n  tasso fra le comunita': min {min(tassi):.1%}  "
              f"media {np.mean(tassi):.1%}  max {max(tassi):.1%}")
        print("  (per confronto: il tasso medio globale e' "
              f"{base:.1%} — variazioni ampie fra comunita' con dimensione")
        print("  simile suggeriscono un effetto di composizione tematica, non solo di scala)")


if __name__ == "__main__":
    main()
