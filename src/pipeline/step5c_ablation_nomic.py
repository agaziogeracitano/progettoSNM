from __future__ import annotations

import json

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, riga, titolo
from step5b_ablation_b3 import (
    B3_LEX, B3_SEM, B3_SEM_PURO, RIPETIZIONI, FOLD, esegui,
)

NOMIC = DATA / "nomic_semantic.parquet"

B3_SEM_NOMIC = ["sim_semantica_op_nomic", "auto_similarita_nomic",
                "ha_auto_similarita"]
B3_SEM_NOMIC_PURO = ["sim_semantica_op_nomic", "auto_similarita_nomic"]


def main() -> None:
    titolo("STEP 5c — ABLATION CON EMBEDDING A CONTESTO LUNGO (nomic)")

    if not NOMIC.exists():
        print(f"\n  MANCA {NOMIC.name}\n")
        print("  Questo script richiede gli embedding calcolati su GPU:")
        print("    1. venv/Scripts/python.exe src/pipeline/export_colab.py")
        print("    2. esegui notebooks/colab_nomic.ipynb su Google Colab")
        print(f"    3. salva il file scaricato in {DATA}")
        print("    4. rilancia questo script")
        raise SystemExit(1)

    blocchi = json.loads((DATA / "blocchi.json").read_text(encoding="utf-8"))
    B1, B2 = blocchi["B1"], blocchi["B2"]
    b3 = json.loads((DATA / "blocco_B3.json").read_text(encoding="utf-8"))
    B3 = b3["colonne"]

    df = pd.read_parquet(DATA / "features.parquet")
    df = df.merge(pd.read_parquet(DATA / b3["file"]), on="unit_id", how="left")
    df = df.merge(pd.read_parquet(NOMIC), on="unit_id", how="left")

    riga("unita' con similarita' nomic",
         int(df["sim_semantica_op_nomic"].notna().sum()), f"su {len(df)}")

    mediana = df["auto_similarita_nomic"].median()
    n_imputate = int(df["auto_similarita_nomic"].isna().sum())
    df["auto_similarita_nomic"] = df["auto_similarita_nomic"].fillna(mediana)
    riga("auto-similarita' imputate alla mediana", n_imputate)

    colonne_tutte = B3 + ["sim_semantica_op_nomic", "auto_similarita_nomic"]
    mancanti = int(df[colonne_tutte].isna().any(axis=1).sum())
    if mancanti:
        riga("unita' con valori mancanti, escluse", mancanti)
        df = df.dropna(subset=colonne_tutte)

    completi = df.groupby("pair_id")["success"].transform("size") == 2
    if (~completi).any():
        riga("unita' spaiate rimosse", int((~completi).sum()))
        df = df[completi]

    riga("coppie appaiate", df["pair_id"].nunique())
    riga("protocollo", f"{RIPETIZIONI} x GroupKFold-{FOLD}", "raggruppato per root")

    # ----------------------------------------------------------------------
    # quanto si somigliano le due misure?
    # ----------------------------------------------------------------------
    titolo("1. LE DUE MISURE DI SIMILARITA' A CONFRONTO")
    a = df["sim_semantica_op"]
    b = df["sim_semantica_op_nomic"]
    print(f"  {'':<26}{'MiniLM (256 tok)':>20}{'nomic (8192 tok)':>20}")
    print("  " + "-" * 66)
    print(f"  {'media':<26}{a.mean():>20.3f}{b.mean():>20.3f}")
    print(f"  {'deviazione standard':<26}{a.std():>20.3f}{b.std():>20.3f}")
    print(f"\n  correlazione di Pearson fra le due: {a.corr(b):.3f}")
    print(f"  correlazione di Spearman:           {a.corr(b, method='spearman'):.3f}")

    # ----------------------------------------------------------------------
    titolo("2. IL SOTTOBLOCCO SEMANTICO, CON E SENZA TRONCAMENTO")
    print(f"  {'config':<30}{'accuratezza':>14}{'AUC':>16}{'vs rif.':>9}{'p':>9}")
    print("  " + "-" * 78)

    tab1, sintesi1 = esegui(df, {
        "B1+B2": B1 + B2,
        "  +sem MiniLM (3)": B1 + B2 + B3_SEM,
        "  +sem nomic (3)": B1 + B2 + B3_SEM_NOMIC,
        "  +sem MiniLM puro (2)": B1 + B2 + B3_SEM_PURO,
        "  +sem nomic puro (2)": B1 + B2 + B3_SEM_NOMIC_PURO,
    }, "B1+B2")

    # ----------------------------------------------------------------------
    titolo("3. IL TEST DECISIVO, RIFATTO SENZA TRONCAMENTO")
    print("  La semantica aggiunge qualcosa SOPRA il lessico, ora che i")
    print("  testi non sono piu' mutilati?\n")
    print(f"  {'config':<30}{'accuratezza':>14}{'AUC':>16}{'vs rif.':>9}{'p':>9}")
    print("  " + "-" * 78)

    tab2, sintesi2 = esegui(df, {
        "B1+B2+B3lex": B1 + B2 + B3_LEX,
        "  +sem MiniLM puro": B1 + B2 + B3_LEX + B3_SEM_PURO,
        "  +sem nomic puro": B1 + B2 + B3_LEX + B3_SEM_NOMIC_PURO,
        "  +entrambi gli encoder": B1 + B2 + B3_LEX + B3_SEM_PURO + B3_SEM_NOMIC_PURO,
    }, "B1+B2+B3lex")

    # ----------------------------------------------------------------------
    titolo("4. SEGNALE UNIVARIATO")
    print("  Frazione di coppie in cui la feature e' piu' alta nel vincente")
    print("  (50% = nessuna associazione).\n")

    vin = df[df["success"] == 1].set_index("pair_id")
    per = df[df["success"] == 0].set_index("pair_id")
    com = vin.index.intersection(per.index)
    for c in ["sim_semantica_op", "sim_semantica_op_nomic",
              "auto_similarita", "auto_similarita_nomic"]:
        x, y = vin.loc[com, c], per.loc[com, c]
        m = x != y
        q = (x > y)[m].mean()
        print(f"  {c:<28}{q:>7.1%}   (n={int(m.sum())})")

    # ----------------------------------------------------------------------
    titolo("SCRITTURA")
    risultati = pd.concat(
        [tab1.assign(confronto="vs B1+B2"), tab2.assign(confronto="vs B1+B2+B3lex")],
        ignore_index=True,
    )
    risultati.to_parquet(DATA / "risultati_nomic.parquet", index=False)
    riga("risultati_nomic.parquet", len(risultati), "righe")

    (DATA / "ablation_nomic.json").write_text(json.dumps({
        "modelli": {
            "riferimento": "all-MiniLM-L6-v2 (256 token)",
            "confronto": "nomic-ai/nomic-embed-text-v1.5 (8192 token)",
        },
        "correlazione_fra_le_due_similarita": {
            "pearson": float(a.corr(b)),
            "spearman": float(a.corr(b, method="spearman")),
        },
        "coppie": int(df["pair_id"].nunique()),
        "confronto_1_vs_B1B2": sintesi1,
        "confronto_2_vs_B1B2B3lex": sintesi2,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    riga("ablation_nomic.json", "riepilogo", "delta, p-value, correlazioni")


if __name__ == "__main__":
    main()
