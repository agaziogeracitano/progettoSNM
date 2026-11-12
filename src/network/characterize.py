from __future__ import annotations

import json
import random

import networkx as nx
import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, riga, titolo

SEED = 42
CAMPIONE_DISTANZE = 400     # coppie per la stima del cammino medio
CAMPIONE_CLUSTERING = 5000  # nodi per la stima del clustering


def stima_clustering(G, rng, k=CAMPIONE_CLUSTERING) -> float:
    nodi = list(G)
    if len(nodi) > k:
        nodi = rng.sample(nodi, k)
    return float(np.mean(list(nx.clustering(G, nodi).values())))


def stima_cammino(G, rng, k=CAMPIONE_DISTANZE) -> float:
    """Cammino medio stimato su sorgenti campionate nella componente gigante."""
    gigante = max(nx.connected_components(G), key=len)
    H = G.subgraph(gigante)
    nodi = list(H)
    tot, n = 0.0, 0
    for s in rng.sample(nodi, min(k, len(nodi))):
        lunghezze = nx.single_source_shortest_path_length(H, s)
        tot += sum(lunghezze.values())
        n += len(lunghezze) - 1
    return tot / n if n else float("nan")


def alpha_power_law(gradi: np.ndarray, k_min: int = 2) -> tuple[float, int]:
    """Stima di massima verosimiglianza dell'esponente per una power-law
    discreta (Clauset et al.), con k_min scelto fra alcuni valori."""
    migliore = (np.nan, k_min, np.inf)
    for km in (1, 2, 3, 5, 10):
        g = gradi[gradi >= km]
        if len(g) < 100:
            continue
        a = 1.0 + len(g) / np.sum(np.log(g / (km - 0.5)))
        # distanza di Kolmogorov-Smirnov fra empirica e teorica
        x = np.sort(g)
        emp = np.arange(len(x)) / len(x)
        teo = 1.0 - (x / (km - 0.5)) ** (1.0 - a)
        ks = float(np.max(np.abs(emp - teo)))
        if ks < migliore[2]:
            migliore = (float(a), km, ks)
    return migliore[0], migliore[1]


def main() -> None:
    titolo("STEP 2c — CARATTERIZZAZIONE DELLA RETE")

    rng = random.Random(SEED)
    archi = pd.read_parquet(DATA / "grafo_archi.parquet")
    G = nx.Graph()
    for u, v, w in archi.itertuples(index=False):
        G.add_edge(u, v, weight=w)

    n, m = G.number_of_nodes(), G.number_of_edges()
    gradi = np.array([d for _, d in G.degree()])

    riga("nodi", n)
    riga("archi", m)
    riga("densita'", f"{2 * m / (n * (n - 1)):.2e}")
    riga("grado medio", f"{gradi.mean():.2f}")
    riga("grado mediano", f"{int(np.median(gradi))}")
    riga("grado massimo", int(gradi.max()))

    # ------------------------------------------------------------------
    titolo("1. DISTRIBUZIONE DEI GRADI")

    alpha, k_min = alpha_power_law(gradi)
    riga("esponente stimato (alpha)", f"{alpha:.2f}", f"per k >= {k_min}")
    frazione_coda = float((gradi >= k_min).mean())
    riga("frazione di nodi nella coda", f"{frazione_coda:.1%}")
    top1 = float(np.sort(gradi)[-max(1, n // 100):].sum() / gradi.sum())
    riga("quota di archi dell'1% piu' connesso", f"{top1:.1%}")

    print("\n  Un alpha fra 2 e 3 e' l'intervallo tipico delle reti sociali")
    print("  reali (preferential attachment). Fuori da li', la coda non e'")
    print("  a legge di potenza e va detto invece che assunto.")

    # ------------------------------------------------------------------
    titolo("2. CONFRONTO CON MODELLI NULLI")

    cc_oss = stima_clustering(G, rng)
    cam_oss = stima_cammino(G, rng)

    print("  generazione di Erdos-Renyi...")
    ER = nx.gnm_random_graph(n, m, seed=SEED)
    cc_er, cam_er = stima_clustering(ER, rng), stima_cammino(ER, rng)

    print("  generazione del configuration model (stessa sequenza di gradi)...")
    CM = nx.configuration_model([int(x) for x in gradi], seed=SEED)
    CM = nx.Graph(CM)          # rimuove multi-archi
    CM.remove_edges_from(nx.selfloop_edges(CM))
    cc_cm, cam_cm = stima_clustering(CM, rng), stima_cammino(CM, rng)

    print()
    print(f"  {'misura':<26}{'osservato':>12}{'Erdos-Renyi':>14}{'config. model':>15}")
    print("  " + "-" * 67)
    print(f"  {'clustering medio':<26}{cc_oss:>12.4f}{cc_er:>14.4f}{cc_cm:>15.4f}")
    print(f"  {'cammino medio':<26}{cam_oss:>12.2f}{cam_er:>14.2f}{cam_cm:>15.2f}")
    print(f"  {'clustering / atteso':<26}{'':>12}{cc_oss / max(cc_er, 1e-9):>13.0f}x"
          f"{cc_oss / max(cc_cm, 1e-9):>14.1f}x")

    piccolo_mondo = (cc_oss / max(cc_er, 1e-9)) / (cam_oss / max(cam_er, 1e-9))
    riga("\ncoefficiente small-world (sigma)", f"{piccolo_mondo:.1f}",
         "small-world se >> 1")

    print("\n  Il confronto che conta e' la colonna del configuration model:")
    print("  a parita' di gradi, il clustering in eccesso e' chiusura triadica,")
    print("  non un artefatto dei nodi molto connessi.")

    # ------------------------------------------------------------------
    titolo("3. OMOFILIA STRUTTURALE (assortativita')")

    r = nx.degree_assortativity_coefficient(G)
    riga("assortativita' di grado", f"{r:+.3f}")
    if r < -0.05:
        print("  Negativa: i nodi molto attivi rispondono soprattutto a nodi")
        print("  poco attivi. E' il segno di una struttura centro-periferia,")
        print("  tipica dei forum: pochi utenti assidui, molti di passaggio.")
    elif r > 0.05:
        print("  Positiva: gli utenti tendono a interagire con propri simili")
        print("  per livello di attivita' (omofilia di grado).")
    else:
        print("  Vicina a zero: nessuna preferenza per grado.")

    # ------------------------------------------------------------------
    titolo("4. STRUTTURA IN COMUNITA'")

    com = nx.community.louvain_communities(G, weight="weight", seed=SEED)
    mod = nx.community.modularity(G, com, weight="weight")
    com_er = nx.community.louvain_communities(ER, seed=SEED)
    mod_er = nx.community.modularity(ER, com_er)

    riga("comunita' trovate", len(com))
    riga("modularita' osservata", f"{mod:.3f}")
    riga("modularita' su Erdos-Renyi", f"{mod_er:.3f}", "(riferimento)")
    dim = sorted((len(c) for c in com), reverse=True)
    riga("dimensione delle 5 maggiori", str(dim[:5]))

    print("\n  Louvain trova comunita' anche in un grafo casuale: e' il difetto")
    print("  noto della massimizzazione della modularita'. Il confronto con la")
    print("  riga di Erdos-Renyi dice se la struttura trovata e' reale.")

    # ------------------------------------------------------------------
    titolo("5. FIGURA")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conteggi = np.bincount(gradi)
    k = np.nonzero(conteggi)[0]
    pk = conteggi[k] / conteggi.sum()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(k, pk, "o", ms=3, alpha=0.6, label="osservata")
    if not np.isnan(alpha):
        kk = k[k >= k_min]
        cost = pk[k >= k_min][0] * (kk[0] ** alpha)
        ax.loglog(kk, cost * kk ** (-alpha), "-", lw=1.5,
                  label=f"power-law $\\alpha$={alpha:.2f}")
    ax.set_xlabel("grado $k$")
    ax.set_ylabel("$P(k)$")
    ax.set_title("Distribuzione dei gradi — rete delle risposte (W_grafo)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DATA / "fig_gradi.png", dpi=150)
    riga("fig_gradi.png", "salvata")

    # ------------------------------------------------------------------
    (DATA / "analisi_rete.json").write_text(json.dumps({
        "nodi": n, "archi": m,
        "grado_medio": float(gradi.mean()), "grado_max": int(gradi.max()),
        "alpha": alpha, "k_min": k_min,
        "clustering": {"osservato": cc_oss, "er": cc_er, "config_model": cc_cm},
        "cammino_medio": {"osservato": cam_oss, "er": cam_er, "config_model": cam_cm},
        "small_world_sigma": piccolo_mondo,
        "assortativita_grado": float(r),
        "modularita": {"osservata": float(mod), "er": float(mod_er),
                       "n_comunita": len(com)},
    }, indent=2), encoding="utf-8")
    riga("analisi_rete.json", "scritto")


if __name__ == "__main__":
    main()
