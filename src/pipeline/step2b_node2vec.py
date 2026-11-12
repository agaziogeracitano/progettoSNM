from __future__ import annotations

import json
import time
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

DIM = 64
LUNGHEZZA = 20
N_WALK = 8
FINESTRA = 3
P_RETURN = 1.0
Q_INOUT = 0.5
EPOCHE = 3
SEED = 42


def main() -> None:
    print("=" * 64)
    print("STEP 2b — node2vec (gensim)")
    print("=" * 64)

    archi = pd.read_parquet(DATA / "grafo_archi.parquet")
    G = nx.Graph()
    for u, v, w in archi.itertuples(index=False):
        G.add_edge(u, v, weight=float(w))
    print(f"  nodi {G.number_of_nodes():,}   archi {G.number_of_edges():,}")
    print(f"  parametri: dim={DIM} p={P_RETURN} q={Q_INOUT} "
          f"walk={N_WALK}x{LUNGHEZZA} finestra={FINESTRA}")

    from node2vec import Node2Vec

    t0 = time.time()
    n2v = Node2Vec(G, dimensions=DIM, walk_length=LUNGHEZZA,
                   num_walks=N_WALK, p=P_RETURN, q=Q_INOUT,
                   weight_key="weight", workers=4, seed=SEED, quiet=False)
    print(f"  cammini generati in {time.time() - t0:.0f}s")

    t0 = time.time()
    modello = n2v.fit(window=FINESTRA, min_count=1, sg=1, negative=5,
                      epochs=EPOCHE, seed=SEED, workers=4)
    print(f"  skip-gram addestrato in {time.time() - t0:.0f}s")

    nodi = list(modello.wv.index_to_key)
    vettori = modello.wv[nodi]
    vettori = vettori / ((vettori ** 2).sum(1, keepdims=True) ** 0.5 + 1e-9)

    colonne = [f"n2v_{i}" for i in range(DIM)]
    out = pd.DataFrame(vettori, columns=colonne)
    out.insert(0, "utente", nodi)
    out.to_parquet(DATA / "node_embeddings.parquet", index=False)

    (DATA / "blocco_B2p.json").write_text(json.dumps({
        "nome": "B2p", "file": "node_embeddings.parquet",
        "chiave": "utente", "colonne": colonne,
        "nota": "node2vec (gensim); si aggancia per sfidante, non per unit_id",
    }, indent=2), encoding="utf-8")

    print(f"\n  node_embeddings.parquet: {len(out):,} righe")

    import numpy as np
    rng = np.random.default_rng(SEED)
    idx = {u: i for i, u in enumerate(nodi)}
    campione = archi.sample(min(3000, len(archi)), random_state=SEED)
    coppie = [(idx[u], idx[v]) for u, v in zip(campione["u"], campione["v"])
              if u in idx and v in idx]
    a = np.array([x for x, _ in coppie])
    b = np.array([y for _, y in coppie])
    sim_vic = float((vettori[a] * vettori[b]).sum(1).mean())
    ra = rng.integers(0, len(nodi), len(coppie))
    rb = rng.integers(0, len(nodi), len(coppie))
    sim_caso = float((vettori[ra] * vettori[rb]).sum(1).mean())

    print(f"  coseno medio fra nodi connessi : {sim_vic:+.3f}")
    print(f"  coseno medio fra nodi a caso   : {sim_caso:+.3f}")
    print(f"  scarto                          : {sim_vic - sim_caso:+.3f}")
    if sim_vic - sim_caso < 0.05:
        print("  ANOMALO: gli embedding non distinguono i vicini dal caso")


if __name__ == "__main__":
    main()
