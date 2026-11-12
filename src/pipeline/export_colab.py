from __future__ import annotations

import json

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, W_MODELLO, riga, titolo


def main() -> None:
    titolo("ESPORTAZIONE PER COLAB")

    units = pd.read_parquet(DATA / "units.parquet")
    units = units[units["split"] == W_MODELLO].reset_index(drop=True)
    threads = pd.read_parquet(DATA / "threads.parquet").set_index("root")

    riga("unita' da esportare", len(units))

    op_testo = [
        (threads.at[r, "op_title"] + "\n" + threads.at[r, "op_text"])
        if r in threads.index else ""
        for r in units["root"]
    ]

    out = pd.DataFrame({
        "unit_id": units["unit_id"],
        "root": units["root"],
        "testi_json": units["testi"],
        "op_testo": op_testo,
    })

    percorso = DATA / "colab_input.parquet"
    out.to_parquet(percorso, index=False, compression="zstd")

    mb = percorso.stat().st_size / 1024 / 1024
    riga("colab_input.parquet", f"{mb:.1f} MB", f"{len(out)} righe")
    riga("thread distinti", units["root"].nunique())
    riga("commenti totali da codificare",
         int(sum(len(json.loads(t)) for t in units["testi"])))

    print("\n  Prossimo passo:")
    print("  1. apri notebooks/colab_nomic.ipynb su Google Colab")
    print("  2. imposta Runtime > Cambia tipo di runtime > GPU (T4)")
    print(f"  3. carica {percorso.name} quando richiesto")
    print("  4. scarica nomic_semantic.parquet in data/ al termine")
    print("  5. esegui src/pipeline/step5c_ablation_nomic.py")


if __name__ == "__main__":
    main()
