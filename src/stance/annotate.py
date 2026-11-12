from __future__ import annotations

import argparse
import json
import random
import sys

import pandas as pd

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, W_MODELLO, riga, titolo

CAMPIONE = DATA / "campione_annotazione.json"
ANNOTAZIONI = DATA / "annotazioni.jsonl"

N_CAMPIONE = 200
SEED = 20260903

ETICHETTE = {
    "1": ("CONCESSIVE", "concede qualcosa di valido all'OP prima di obiettare"),
    "2": ("DIRECT", "contraddice frontalmente, nessuna concessione"),
    "3": ("CLARIFYING", "chiede chiarimenti o ridefinisce i termini"),
    "4": ("HOSTILE", "disaccordo con tono sprezzante o ostile verso l'OP"),
    "5": ("AGREE", "in realta' e' d'accordo con l'OP"),
    "6": ("OTHER", "nessuna delle precedenti"),
}

REGOLE = """
  Criteri (gli stessi dati all'LLM, per rendere il confronto equo):
    - giudica il tono verso l'OP, non verso terzi
    - un'apertura cortese NON basta per CONCESSIVE: serve una concessione vera
    - se ci sono elementi sia concessivi sia ostili, scegli quello dominante
"""


def estrai_campione() -> list[str]:
    """Campione stratificato per esito, estratto una volta sola e congelato."""
    if CAMPIONE.exists():
        return json.loads(CAMPIONE.read_text(encoding="utf-8"))["unit_ids"]

    units = pd.read_parquet(DATA / "units.parquet")
    units = units[units["split"] == W_MODELLO]

    rng = random.Random(SEED)
    scelti: list[str] = []
    for esito in (1, 0):
        pool = sorted(units.loc[units["success"] == esito, "unit_id"])
        scelti += rng.sample(pool, N_CAMPIONE // 2)
    rng.shuffle(scelti)      # ordine mescolato: non deve trasparire l'esito

    CAMPIONE.write_text(json.dumps({
        "seed": SEED, "n": len(scelti),
        "nota": "stratificato per esito, ordine mescolato, congelato",
        "unit_ids": scelti,
    }, indent=2), encoding="utf-8")
    return scelti


def mostra(n: int, tot: int, op_titolo: str, op_testo: str, testo: str) -> None:
    print("\n" * 2 + "=" * 74)
    print(f"  {n} di {tot}")
    print("=" * 74)
    print(f"\n  POST DELL'OP — {op_titolo.strip()[:150]}\n")
    corpo = " ".join(op_testo.split())
    print("  " + (corpo[:700] + " [...]" if len(corpo) > 700 else corpo))
    print("\n" + "-" * 74)
    print("  RISPOSTA DELLO SFIDANTE\n")
    sfid = " ".join(testo.split())
    print("  " + (sfid[:2500] + " [...]" if len(sfid) > 2500 else sfid))
    print("\n" + "-" * 74)
    print("  " + "   ".join(f"{k}={v[0]}" for k, v in ETICHETTE.items()))
    print("  ?=definizioni   s=salta   q=esci")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stato", action="store_true", help="solo il conteggio")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    titolo("ANNOTAZIONE MANUALE — validazione dello stance")

    ids = estrai_campione()
    units = pd.read_parquet(DATA / "units.parquet").set_index("unit_id")
    threads = pd.read_parquet(DATA / "threads.parquet").set_index("root")

    fatte: dict[str, str] = {}
    if ANNOTAZIONI.exists():
        for r in ANNOTAZIONI.read_text(encoding="utf-8").splitlines():
            if r.strip():
                d = json.loads(r)
                fatte[d["unit_id"]] = d["stance"]

    restanti = [i for i in ids if i not in fatte]
    riga("campione", len(ids))
    riga("gia' annotate", len(fatte))
    riga("restanti", len(restanti))

    if args.stato or not restanti:
        if not restanti:
            print("\n  Campione completo. Ora: venv/Scripts/python.exe src/stance/validate.py")
        return

    print(REGOLE)
    print("  Il campione e' congelato e le tue risposte sono salvate a ogni")
    print("  passo: puoi uscire con 'q' e riprendere quando vuoi.\n")
    input("  Invio per iniziare...")

    with open(ANNOTAZIONI, "a", encoding="utf-8") as out:
        for k, uid in enumerate(restanti, 1):
            u = units.loc[uid]
            t = threads.loc[u["root"]] if u["root"] in threads.index else None
            mostra(len(fatte) + k, len(ids),
                   t["op_title"] if t is not None else "",
                   t["op_text"] if t is not None else "",
                   " ".join(json.loads(u["testi"])))

            while True:
                scelta = input("\n  > ").strip().lower()
                if scelta == "q":
                    print(f"\n  Salvate {len(fatte) + k - 1} annotazioni. "
                          "Rilancia lo script per riprendere.\n")
                    return
                if scelta == "?":
                    print()
                    for c, (nome, desc) in ETICHETTE.items():
                        print(f"    {c} {nome:<12} {desc}")
                    print(REGOLE)
                    continue
                if scelta == "s":
                    out.write(json.dumps({"unit_id": uid, "stance": "SALTATA"}) + "\n")
                    out.flush()
                    break
                if scelta in ETICHETTE:
                    out.write(json.dumps(
                        {"unit_id": uid, "stance": ETICHETTE[scelta][0]}) + "\n")
                    out.flush()
                    break
                print("  tasto non valido")

    print("\n  Campione completo. Ora: venv/Scripts/python.exe src/stance/validate.py\n")


if __name__ == "__main__":
    main()
