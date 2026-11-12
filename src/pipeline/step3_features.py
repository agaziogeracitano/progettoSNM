from __future__ import annotations

import json

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, URL_PATTERN, W_MODELLO, riga, titolo

# colonne prodotte da step2_graph.py che formano il blocco B2
COLONNE_RETE = [
    "degree", "strength", "clustering", "pagerank", "betweenness",
    "partecipazione", "n_comunita_vicine", "n_thread", "n_commenti",
    "delta_vinti", "tentativi", "tasso_delta",
]

PAROLE_CONNETTIVE = (
    "because", "however", "although", "therefore", "actually",
    "evidence", "study", "data", "example", "consider",
)


def feature_superficiali(r: pd.Series, t_thread: float | None) -> dict:
    """B1 — proprieta' del testo dello sfidante, senza semantica."""
    testi = json.loads(r["testi"])
    stamps = [t for t in json.loads(r["timestamps"]) if t is not None]
    prof = json.loads(r["profondita"])

    parole = [len(t.split()) for t in testi]
    unito = "\n".join(testi)
    minuscolo = unito.lower()
    righe = unito.split("\n")

    primo = min(stamps) if stamps else None
    ts = pd.to_datetime(primo, unit="s", utc=True) if primo else None

    return {
        # volume
        "n_utterance": len(testi),          # confondente residuo, vedi B1-
        "n_parole": sum(parole),
        "n_parole_media": float(np.mean(parole)) if parole else 0.0,
        "n_parole_max": max(parole) if parole else 0,
        # fonti e struttura
        "n_link": len(URL_PATTERN.findall(unito)),
        "n_citazioni": sum(1 for x in righe if x.lstrip().startswith("&gt;")
                           or x.lstrip().startswith(">")),
        "n_paragrafi": sum(1 for x in righe if x.strip()),
        "n_elenchi": sum(1 for x in righe
                         if x.lstrip()[:2] in ("* ", "- ") or x.lstrip()[:3] == "1. "),
        "n_domande": unito.count("?"),
        "n_connettivi": sum(minuscolo.count(p) for p in PAROLE_CONNETTIVE),
        # posizione nella discussione
        "profondita_prima": prof[0] if prof else 0,
        "profondita_media": float(np.mean(prof)) if prof else 0.0,
        "ritardo_ore": ((primo - t_thread) / 3600.0
                        if primo and t_thread else 0.0),
        # tempo
        "ora": ts.hour if ts is not None else 0,
        "giorno_settimana": ts.dayofweek if ts is not None else 0,
    }


def main() -> None:
    titolo("STEP 3 — FEATURE B1 (superficiali) + B2 (rete)")

    units = pd.read_parquet(DATA / "units.parquet")
    units = units[units["split"] == W_MODELLO].reset_index(drop=True)
    threads = pd.read_parquet(DATA / "threads.parquet")
    nodi = pd.read_parquet(DATA / "node_features.parquet")

    riga("unita'", len(units))
    riga("utenti nel grafo (W_grafo)", len(nodi))

    # ----------------------------------------------------------------------
    # B1
    # ----------------------------------------------------------------------
    titolo("B1 — SUPERFICIALI")

    t_thread = threads.set_index("root")["timestamp"].to_dict()
    b1 = pd.DataFrame([
        feature_superficiali(r, t_thread.get(r["root"]))
        for _, r in units.iterrows()
    ])
    colonne_b1 = list(b1.columns)
    riga("feature B1", len(colonne_b1))

    # ----------------------------------------------------------------------
    # B2
    # ----------------------------------------------------------------------
    titolo("B2 — RETE")

    base = units[["unit_id", "pair_id", "root", "success", "sfidante", "split"]]
    df = pd.concat([base.reset_index(drop=True), b1], axis=1)

    df = df.merge(
        nodi.rename(columns={"utente": "sfidante"})[["sfidante"] + COLONNE_RETE],
        on="sfidante", how="left",
    )
    df["noto_nel_grafo"] = df["degree"].notna().astype(int)
    df[COLONNE_RETE] = df[COLONNE_RETE].fillna(0.0)
    df.loc[df["noto_nel_grafo"] == 0, "tasso_delta"] = 0.5

    colonne_b2 = COLONNE_RETE + ["noto_nel_grafo"]
    riga("feature B2", len(colonne_b2))

    riga("copertura (sfidante noto nel grafo)", f"{df['noto_nel_grafo'].mean():.1%}")
    per_coppia = df.groupby("pair_id")["noto_nel_grafo"].sum()
    riga("coppie con entrambi noti", int((per_coppia == 2).sum()),
         f"({(per_coppia == 2).mean():.1%})")

    # ----------------------------------------------------------------------
    # controlli
    # ----------------------------------------------------------------------
    titolo("CONTROLLI")

    coppie = df.groupby("pair_id")["success"].agg(["size", "sum"])
    ok = ((coppie["size"] == 2) & (coppie["sum"] == 1)).all()
    print(f"  [{'v' if ok else 'X'}] ogni coppia ha 2 unita', una vincente")

    nan = df[colonne_b1 + colonne_b2].isna().sum().sum()
    print(f"  [{'v' if nan == 0 else 'X'}] valori mancanti: {nan}")

    inf = np.isinf(df[colonne_b1 + colonne_b2].to_numpy(dtype=float)).sum()
    print(f"  [{'v' if inf == 0 else 'X'}] valori infiniti: {inf}")

    # ----------------------------------------------------------------------
    # scrittura + colpo d'occhio sulle differenze appaiate
    # ----------------------------------------------------------------------
    df.to_parquet(DATA / "features.parquet", index=False)
    with open(DATA / "blocchi.json", "w", encoding="utf-8") as f:
        json.dump({"B1": colonne_b1, "B2": colonne_b2}, f, indent=2)

    titolo("SEGNALE GREZZO — differenze dentro la coppia")
    print("  Frazione di coppie in cui la feature e' PIU' ALTA nel vincente.")
    print("  50% = nessun segnale. Nessuna correzione per test multipli qui:")
    print("  e' un colpo d'occhio, non un risultato.\n")

    vin = df[df["success"] == 1].set_index("pair_id")
    per = df[df["success"] == 0].set_index("pair_id")
    comuni = vin.index.intersection(per.index)

    esiti = []
    for c in colonne_b1 + colonne_b2:
        a, b = vin.loc[comuni, c], per.loc[comuni, c]
        diverse = (a != b)
        if diverse.sum() == 0:
            continue
        esiti.append((c, (a > b)[diverse].mean(), int(diverse.sum())))

    for nome, q, n in sorted(esiti, key=lambda x: -abs(x[1] - 0.5))[:12]:
        barra = "#" * int(abs(q - 0.5) * 100)
        print(f"  {nome:<22}{q:>7.1%}   (n={n:>5})  {barra}")

    riga("\nfeatures.parquet", len(df), "righe")


if __name__ == "__main__":
    main()
