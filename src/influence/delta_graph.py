from __future__ import annotations

import json
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, e_escluso, riga, titolo


def main() -> None:
    titolo("STEP 6 — GRAFO DEI DELTA")

    units = pd.read_parquet(DATA / "units.parquet")
    threads = pd.read_parquet(DATA / "threads.parquet").set_index("root")

    vincenti = units[units["success"] == 1]
    riga("unita' vincenti (delta osservati)", len(vincenti))

    pesi: Counter = Counter()
    saltati = 0
    for root, sfidante in zip(vincenti["root"], vincenti["sfidante"]):
        if root not in threads.index:
            saltati += 1
            continue
        op = threads.at[root, "op"]
        if e_escluso(op) or e_escluso(sfidante) or op == sfidante:
            saltati += 1
            continue
        pesi[(sfidante, op)] += 1        # sfidante PERSUADE op

    riga("scartati (OP mancante/bot/self)", saltati)

    archi = pd.DataFrame([(a, b, w) for (a, b), w in pesi.items()],
                         columns=["persuasore", "persuaso", "peso"])
    archi.to_parquet(DATA / "grafo_delta.parquet", index=False)

    D = nx.DiGraph()
    for a, b, w in archi.itertuples(index=False):
        D.add_edge(a, b, weight=w)

    n, m = D.number_of_nodes(), D.number_of_edges()
    riga("nodi", n)
    riga("archi", m)
    riga("archi ripetuti (stessa coppia >1 delta)", int((archi["peso"] > 1).sum()))

    # ------------------------------------------------------------------
    titolo("1. CHI PERSUADE E CHI VIENE PERSUASO")

    out_deg = dict(D.out_degree(weight="weight"))     # delta dati
    in_deg = dict(D.in_degree(weight="weight"))       # delta ricevuti

    persuasori = np.array([v for v in out_deg.values() if v > 0])
    persuasi = np.array([v for v in in_deg.values() if v > 0])

    riga("persone che hanno persuaso almeno una volta", len(persuasori))
    riga("persone persuase almeno una volta", len(persuasi))
    riga("delta totali", int(persuasori.sum()))
    print()
    riga("delta dati — media / max", f"{persuasori.mean():.2f} / {int(persuasori.max())}")
    riga("delta ricevuti — media / max", f"{persuasi.mean():.2f} / {int(persuasi.max())}")

    ordinati = np.sort(persuasori)[::-1]
    quota = ordinati[:max(1, len(ordinati) // 100)].sum() / ordinati.sum()
    riga("quota di delta dell'1% piu' persuasivo", f"{quota:.1%}")
    riga("persone con un solo delta dato",
         f"{(persuasori == 1).mean():.1%}")

    print("\n  Se pochissimi utenti concentrano molti delta, la persuasione e'")
    print("  un'abilita' concentrata; se quasi tutti ne hanno uno solo, e'")
    print("  un evento occasionale. E' la differenza fra cercare influencer")
    print("  e cercare occasioni.")

    # ------------------------------------------------------------------
    titolo("2. STRUTTURA")

    deboli = sorted(nx.weakly_connected_components(D), key=len, reverse=True)
    riga("componenti deboli", len(deboli))
    riga("componente debole gigante", len(deboli[0]),
         f"({len(deboli[0]) / n:.1%})")

    forti = sorted(nx.strongly_connected_components(D), key=len, reverse=True)
    riga("componenti forti", len(forti))
    riga("componente forte gigante", len(forti[0]))

    reciproci = sum(1 for a, b in D.edges() if D.has_edge(b, a))
    riga("archi reciproci", reciproci,
         f"({reciproci / m:.2%})  = persuasione a due sensi")

    dag = nx.is_directed_acyclic_graph(D)
    riga("il grafo e' aciclico", "si" if dag else "no")

    # sovrapposizione con la rete delle risposte
    try:
        reply = pd.read_parquet(DATA / "grafo_archi.parquet")
        coppie_reply = {tuple(sorted((u, v))) for u, v, _ in reply.itertuples(index=False)}
        dentro = sum(1 for a, b in D.edges() if tuple(sorted((a, b))) in coppie_reply)
        riga("archi delta presenti anche in W_grafo", dentro,
             f"({dentro / m:.1%})")
        print("\n  La quota bassa e' attesa: W_grafo copre solo la prima meta'")
        print("  temporale, mentre i delta qui sono su tutto il corpus.")
    except FileNotFoundError:
        pass

    # ------------------------------------------------------------------
    titolo("3. I PIU' PERSUASIVI")

    top = sorted(out_deg.items(), key=lambda x: -x[1])[:10]
    print(f"  {'utente':<26}{'delta dati':>12}{'persone distinte':>18}")
    print("  " + "-" * 56)
    for u, d in top:
        print(f"  {u:<26}{int(d):>12}{D.out_degree(u):>18}")

    # ------------------------------------------------------------------
    (DATA / "analisi_delta.json").write_text(json.dumps({
        "nodi": n, "archi": m, "delta_totali": int(persuasori.sum()),
        "persuasori": len(persuasori), "persuasi": len(persuasi),
        "max_delta_dati": int(persuasori.max()),
        "quota_top1pct": float(quota),
        "frazione_un_solo_delta": float((persuasori == 1).mean()),
        "componente_debole_gigante": len(deboli[0]),
        "componente_forte_gigante": len(forti[0]),
        "archi_reciproci": reciproci,
        "aciclico": bool(dag),
    }, indent=2), encoding="utf-8")
    riga("\nanalisi_delta.json", "scritto")

    # ------------------------------------------------------------------
    titolo("VERDETTO PER L'INFLUENCE MAXIMIZATION")
    if len(deboli[0]) < 0.3 * n or m < 1000:
        print("  Il grafo dei delta e' troppo frammentato per far girare IM")
        print("  direttamente: la diffusione si fermerebbe dopo un passo.")
        print("  IM andra' eseguita sulla rete delle risposte, usando i delta")
        print("  per STIMARE le probabilita' sugli archi.")
    else:
        print("  Componente gigante sufficiente: IM puo' girare su questo grafo.")


if __name__ == "__main__":
    main()
