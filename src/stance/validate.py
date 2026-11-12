from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, riga, titolo

SOGLIA_BUONA = 0.60
SOGLIA_MINIMA = 0.40


def main() -> None:
    titolo("VALIDAZIONE DELLO STANCE — Cohen's kappa")

    umane = pd.read_json(DATA / "annotazioni.jsonl", lines=True)
    umane = umane.drop_duplicates("unit_id", keep="last")
    saltate = int((umane["stance"] == "SALTATA").sum())
    umane = umane[umane["stance"] != "SALTATA"]

    llm = pd.read_parquet(DATA / "stance.parquet").drop_duplicates("unit_id")
    llm = llm[llm["stance"] != "FALLITO"]

    d = umane.merge(llm, on="unit_id", suffixes=("_umana", "_llm"))

    riga("annotate a mano", len(umane) + saltate)
    riga("  saltate (ambigue)", saltate)
    riga("classificate dall'LLM", len(llm))
    riga("confrontabili", len(d))

    if len(d) < 50:
        print("\n  Troppo poche per un kappa affidabile.")
        print("  Servono almeno 50 unita' presenti in ENTRAMBE le fonti:")
        print("  completa l'annotazione e attendi la fine di stance/classify.py.")
        return

    a, b = d["stance_umana"], d["stance_llm"]
    etichette = sorted(set(a) | set(b))

    kappa = cohen_kappa_score(a, b, labels=etichette)
    accordo = float((a == b).mean())

    titolo("RISULTATO")
    riga("accordo grezzo", f"{accordo:.1%}")
    riga("Cohen's kappa", f"{kappa:.3f}")

    # ------------------------------------------------------------------
    titolo("MATRICE DI CONFUSIONE")
    m = confusion_matrix(a, b, labels=etichette)
    print(f"  righe = umana, colonne = LLM\n")
    print("  " + " " * 13 + "".join(f"{e[:6]:>8}" for e in etichette))
    for i, e in enumerate(etichette):
        print(f"  {e:<13}" + "".join(f"{v:>8}" for v in m[i]))

    # ------------------------------------------------------------------
    titolo("ACCORDO PER CLASSE")
    print(f"  {'classe':<14}{'n umana':>9}{'n LLM':>8}{'concordi':>10}{'recall':>9}")
    print("  " + "-" * 50)
    per_classe = {}
    for e in etichette:
        nu = int((a == e).sum())
        nl = int((b == e).sum())
        conc = int(((a == e) & (b == e)).sum())
        rec = conc / nu if nu else float("nan")
        per_classe[e] = {"n_umana": nu, "n_llm": nl, "concordi": conc,
                         "recall": None if nu == 0 else round(rec, 3)}
        print(f"  {e:<14}{nu:>9}{nl:>8}{conc:>10}"
              f"{'  —' if not nu else f'{rec:>9.1%}'}")

    # ------------------------------------------------------------------
    titolo("VERDETTO")
    if kappa > SOGLIA_BUONA:
        verdetto = "PROCEDI"
        print(f"  kappa = {kappa:.3f} > {SOGLIA_BUONA}")
        print("  Il blocco M entra nel modello.")
    elif kappa >= SOGLIA_MINIMA:
        verdetto = "PROCEDI CON RISERVA"
        print(f"  {SOGLIA_MINIMA} <= kappa = {kappa:.3f} <= {SOGLIA_BUONA}")
        print("  Il blocco M entra nel modello, MA l'accordo moderato va")
        print("  dichiarato come limite forte nella relazione: qualunque")
        print("  guadagno di M va letto con questa incertezza sopra.")
    else:
        verdetto = "ESCLUDI M"
        print(f"  kappa = {kappa:.3f} < {SOGLIA_MINIMA}")
        print("  Lo stance detection zero-shot non produce segnale affidabile")
        print("  su questo task. Il blocco M ESCE dal modello e si riporta")
        print("  come risultato negativo di RQ3 — che era previsto in §8 come")
        print("  esito possibile, quindi e' un risultato, non un fallimento.")
        print("\n  Operativamente: cancella data/blocco_M.json cosi' step5_train.py")
        print("  non lo aggancia, e riporta questo numero nella relazione.")

    (DATA / "validazione.json").write_text(json.dumps({
        "n_confrontabili": len(d),
        "n_saltate": saltate,
        "accordo_grezzo": round(accordo, 4),
        "kappa": round(float(kappa), 4),
        "verdetto": verdetto,
        "per_classe": per_classe,
    }, indent=2), encoding="utf-8")
    riga("\nvalidazione.json", "scritto")


if __name__ == "__main__":
    main()
