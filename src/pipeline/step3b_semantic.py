from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, W_MODELLO, riga, titolo

MODELLO = "all-MiniLM-L6-v2"     # ~90 MB, veloce su CPU, piu' che adeguato
STOP = frozenset(ENGLISH_STOP_WORDS)


def parole(testo: str) -> list[str]:
    return [p for p in "".join(
        c.lower() if c.isalnum() or c.isspace() else " " for c in testo
    ).split() if len(p) > 1]


def interplay(testo_sfidante: str, testo_op: str) -> dict:
    """Sovrapposizione lessicale fra la replica e il post dell'OP."""
    ps, po = parole(testo_sfidante), parole(testo_op)
    if not ps or not po:
        return {k: 0.0 for k in (
            "ip_contenuto_coperto", "ip_jaccard", "ip_stop_ripreso",
            "ip_parole_nuove", "diversita_lessicale")}

    cs, co = {p for p in ps if p not in STOP}, {p for p in po if p not in STOP}
    ss, so = {p for p in ps if p in STOP}, {p for p in po if p in STOP}
    comuni = cs & co

    return {
        # quanto dell'OP viene toccato dallo sfidante
        "ip_contenuto_coperto": len(comuni) / len(co) if co else 0.0,
        "ip_jaccard": len(comuni) / len(cs | co) if (cs | co) else 0.0,
        # allineamento di stile, non di argomento
        "ip_stop_ripreso": len(ss & so) / len(ss) if ss else 0.0,
        # termini che lo sfidante porta e l'OP non aveva
        "ip_parole_nuove": len(cs - co) / len(cs) if cs else 0.0,
        "diversita_lessicale": len(set(ps)) / len(ps),
    }


def main() -> None:
    titolo("STEP 3b — FEATURE SEMANTICHE (B3)")

    from sentence_transformers import SentenceTransformer

    units = pd.read_parquet(DATA / "units.parquet")
    units = units[units["split"] == W_MODELLO].reset_index(drop=True)
    threads = pd.read_parquet(DATA / "threads.parquet").set_index("root")

    riga("unita'", len(units))
    riga("modello", MODELLO)

    testi_sfidante = [" ".join(json.loads(t)) for t in units["testi"]]
    testi_op = [
        (threads.at[r, "op_title"] + "\n" + threads.at[r, "op_text"])
        if r in threads.index else ""
        for r in units["root"]
    ]

    # ----------------------------------------------------------------------
    titolo("1. INTERPLAY LESSICALE")
    ip = pd.DataFrame([interplay(a, b) for a, b in zip(testi_sfidante, testi_op)])
    riga("feature di interplay", len(ip.columns))

    # ----------------------------------------------------------------------
    titolo("2. EMBEDDING")
    print(f"  caricamento di {MODELLO} (prima volta: ~90 MB di download)...")
    modello = SentenceTransformer(MODELLO)

    print("  embedding dello sfidante...")
    e_sfid = modello.encode(testi_sfidante, batch_size=64, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=True)
    print("  embedding dei post dell'OP...")
    root_unici = list(dict.fromkeys(units["root"]))
    testi_root = [(threads.at[r, "op_title"] + "\n" + threads.at[r, "op_text"])
                  if r in threads.index else "" for r in root_unici]
    e_root = modello.encode(testi_root, batch_size=64, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=True)
    indice = {r: i for i, r in enumerate(root_unici)}
    e_op = e_root[[indice[r] for r in units["root"]]]

    # normalizzati: il prodotto scalare E' il coseno
    sim_op = (e_sfid * e_op).sum(axis=1)

    print("  auto-similarita' fra utterance successive...")
    auto = []
    for t in units["testi"]:
        parti = json.loads(t)
        if len(parti) < 2:
            auto.append(np.nan)
            continue
        e = modello.encode(parti, convert_to_numpy=True, normalize_embeddings=True)
        auto.append(float(np.mean([
            float(e[i] @ e[i + 1]) for i in range(len(e) - 1)
        ])))
    auto = pd.Series(auto)
    b3 = ip.assign(
        sim_semantica_op=sim_op,
        auto_similarita=auto.fillna(auto.median() if auto.notna().any() else 0.0),
        ha_auto_similarita=auto.notna().astype(int),
    )

    # ----------------------------------------------------------------------
    titolo("3. SCRITTURA")
    out = pd.concat([units[["unit_id"]], b3], axis=1)
    out.to_parquet(DATA / "semantic.parquet", index=False)
    (DATA / "blocco_B3.json").write_text(json.dumps({
        "nome": "B3", "file": "semantic.parquet",
        "colonne": list(b3.columns),
    }, indent=2), encoding="utf-8")
    riga("semantic.parquet", len(out), "righe")
    riga("feature B3", len(b3.columns))

    # ----------------------------------------------------------------------
    titolo("SEGNALE GREZZO — differenze dentro la coppia")
    print("  Frazione di coppie in cui la feature e' PIU' ALTA nel vincente.\n")
    d = pd.concat([units[["pair_id", "success"]], b3], axis=1)
    vin = d[d["success"] == 1].set_index("pair_id")
    per = d[d["success"] == 0].set_index("pair_id")
    com = vin.index.intersection(per.index)
    for c in b3.columns:
        a, b = vin.loc[com, c], per.loc[com, c]
        m = a != b
        if m.sum() < 30:
            continue
        q = (a > b)[m].mean()
        print(f"  {c:<24}{q:>7.1%}   (n={int(m.sum()):>5})  "
              f"{'#' * int(abs(q - 0.5) * 100)}")


if __name__ == "__main__":
    main()
