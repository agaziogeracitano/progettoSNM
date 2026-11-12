from __future__ import annotations

from collections import Counter, defaultdict

import networkx as nx
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    DATA, W_GRAFO, W_MODELLO, campo_padre, carica_utterance, e_escluso,
    mappa_op, nome_autore, riga, titolo,
)

PIVOT_BETWEENNESS = 400
SEED = 42


def main() -> None:
    titolo("STEP 2 — GRAFO DI INTERAZIONE SU W_grafo")

    units = pd.read_parquet(DATA / "units.parquet")
    threads = pd.read_parquet(DATA / "threads.parquet")
    grafo_roots = set(threads.loc[threads["split"] == W_GRAFO, "root"])

    print("\nlettura del corpus...")
    utt = carica_utterance()
    op = mappa_op()

    # ----------------------------------------------------------------------
    # 1. costruzione del grafo
    # ----------------------------------------------------------------------
    titolo("1. COSTRUZIONE")

    pesi: Counter = Counter()
    thread_per_utente: dict[str, set] = defaultdict(set)
    commenti: Counter = Counter()

    saltate_no_padre = saltate_orfane = saltate_escluse = saltate_self = 0

    for u in utt.values():
        if u["root"] not in grafo_roots:
            continue
        autore = nome_autore(u)
        if not e_escluso(autore):
            thread_per_utente[autore].add(u["root"])
            commenti[autore] += 1

        padre_id = campo_padre(u)
        if not padre_id:
            saltate_no_padre += 1
            continue
        padre = utt.get(padre_id)
        if padre is None:
            saltate_orfane += 1
            continue
        b = nome_autore(padre)
        if e_escluso(autore) or e_escluso(b):
            saltate_escluse += 1
            continue
        if autore == b:
            saltate_self += 1
            continue
        pesi[tuple(sorted((autore, b)))] += 1

    G = nx.Graph()
    for (a, b), w in pesi.items():
        G.add_edge(a, b, weight=w)

    riga("thread usati (W_grafo)", len(grafo_roots))
    riga("commenti di primo livello", saltate_no_padre)
    riga("reply orfane (padre fuori corpus)", saltate_orfane)
    riga("scartate (bot / [deleted])", saltate_escluse)
    riga("autorisposte", saltate_self)
    print()
    riga("nodi", G.number_of_nodes())
    riga("archi", G.number_of_edges())
    componenti = sorted(nx.connected_components(G), key=len, reverse=True)
    riga("componenti connesse", len(componenti))
    riga("componente gigante", len(componenti[0]),
         f"({len(componenti[0]) / G.number_of_nodes():.1%})")

    # ----------------------------------------------------------------------
    # 2. delta storici
    # ----------------------------------------------------------------------
    titolo("2. REPUTAZIONE PERSUASIVA PREGRESSA")

    grafo_units = units[units["split"] == W_GRAFO]
    delta_vinti = Counter(grafo_units.loc[grafo_units["success"] == 1, "sfidante"])
    tentativi = Counter(grafo_units["sfidante"])

    riga("utenti con >=1 delta in W_grafo", sum(1 for v in delta_vinti.values() if v))
    riga("delta totali in W_grafo", sum(delta_vinti.values()))

    # ----------------------------------------------------------------------
    # 3. centralita' e struttura
    # ----------------------------------------------------------------------
    titolo("3. CENTRALITA' E STRUTTURA")

    print("  degree / strength...")
    degree = dict(G.degree())
    strength = dict(G.degree(weight="weight"))

    print("  clustering (chiusura triadica)...")
    clustering = nx.clustering(G)

    print("  pagerank...")
    pagerank = nx.pagerank(G, weight="weight")

    print(f"  betweenness approssimata ({PIVOT_BETWEENNESS} pivot)...")
    betweenness = nx.betweenness_centrality(G, k=PIVOT_BETWEENNESS, seed=SEED)

    print("  community detection (Louvain)...")
    comunita = nx.community.louvain_communities(G, weight="weight", seed=SEED)
    di_chi = {n: i for i, c in enumerate(comunita) for n in c}
    riga("comunita' trovate", len(comunita))
    riga("comunita' con >=10 nodi", sum(1 for c in comunita if len(c) >= 10))

    print("  coefficiente di partecipazione (ruolo di ponte)...")
    partecipazione, n_comunita_vicine = {}, {}
    for n in G:
        vicini = list(G[n])
        if not vicini:
            partecipazione[n], n_comunita_vicine[n] = 0.0, 0
            continue
        conta = Counter(di_chi[v] for v in vicini)
        k = len(vicini)
        partecipazione[n] = 1.0 - sum((c / k) ** 2 for c in conta.values())
        n_comunita_vicine[n] = len(conta)

    # ----------------------------------------------------------------------
    # 4. scrittura
    # ----------------------------------------------------------------------
    titolo("4. SCRITTURA")

    righe = []
    for n in G:
        t = tentativi.get(n, 0)
        d = delta_vinti.get(n, 0)
        righe.append({
            "utente": n,
            "degree": degree[n],
            "strength": strength[n],
            "clustering": clustering[n],
            "pagerank": pagerank[n],
            "betweenness": betweenness[n],
            "partecipazione": partecipazione[n],
            "n_comunita_vicine": n_comunita_vicine[n],
            "n_thread": len(thread_per_utente.get(n, ())),
            "n_commenti": commenti.get(n, 0),
            "delta_vinti": d,
            "tentativi": t,
            "tasso_delta": (d + 0.5 * 2) / (t + 2) if t else 0.5,
        })

    nodi = pd.DataFrame(righe)
    nodi.to_parquet(DATA / "node_features.parquet", index=False)
    riga("node_features.parquet", len(nodi), "righe")

    archi = pd.DataFrame(
        [(a, b, w) for (a, b), w in pesi.items()],
        columns=["u", "v", "peso"],
    )
    archi.to_parquet(DATA / "grafo_archi.parquet", index=False)
    riga("grafo_archi.parquet", len(archi), "archi")

    # ----------------------------------------------------------------------
    # 5. copertura sul test set — e' la premessa di RQ1
    # ----------------------------------------------------------------------
    titolo("5. COPERTURA SU W_modello")

    noti = set(nodi["utente"])
    test = units[units["split"] == W_MODELLO]
    coperte = test["sfidante"].isin(noti)

    riga("unita' in W_modello", len(test))
    riga("sfidante noto in W_grafo", int(coperte.sum()),
         f"({coperte.mean():.1%})")
    for s in (1, 0):
        m = test["success"] == s
        riga(f"  fra le {'vincenti' if s else 'perdenti'}",
             int(coperte[m].sum()), f"({coperte[m].mean():.1%})")

    per_coppia = test.assign(noto=coperte).groupby("pair_id")["noto"].sum()
    riga("coppie con entrambi noti", int((per_coppia == 2).sum()),
         f"({(per_coppia == 2).mean():.1%} di {len(per_coppia)})")

    print("\n  Le due righe per esito devono essere quasi uguali: se lo sono,")
    print("  'essere gia' noto alla rete' non predice il delta e non e' un")
    print("  canale di leakage. Il divario fra coperte e non coperte e' invece")
    print("  la misura diretta del valore dell'informazione di rete (RQ1).")


if __name__ == "__main__":
    main()
