from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    DATA, DELTA_PATTERN, FRAZIONE_GRAFO, W_GRAFO, W_MODELLO,
    carica_conversazioni, carica_utterance, campo_padre, controllo,
    e_escluso, mappa_op, nome_autore, riga, titolo,
)


def profondita(utt: dict[str, dict]) -> dict[str, int]:
    """Distanza di ogni utterance dalla radice del thread, con memoizzazione."""
    prof: dict[str, int] = {}

    def calcola(uid: str) -> int:
        if uid in prof:
            return prof[uid]
        catena = []
        corrente = uid
        while corrente is not None and corrente not in prof:
            u = utt.get(corrente)
            if u is None:                       # padre fuori dal corpus
                prof[corrente] = 0
                break
            padre = campo_padre(u)
            if padre is None:                   # e' la radice
                prof[corrente] = 0
                break
            catena.append(corrente)
            corrente = padre
        base = prof.get(corrente, 0)
        for uid_c in reversed(catena):
            base += 1
            prof[uid_c] = base
        return prof.get(uid, 0)

    for uid in utt:
        calcola(uid)
    return prof


def main() -> None:
    titolo("STEP 1 — COSTRUZIONE DELLE UNITA' DI PERCORSO")

    print("\nlettura del corpus...")
    utt = carica_utterance()
    conv = carica_conversazioni()
    op = mappa_op(conv)
    riga("utterance lette", len(utt))
    riga("thread (conversazioni)", len(conv))

    # ----------------------------------------------------------------------
    # 1. raggruppamento per (root, pair_id, success)
    # ----------------------------------------------------------------------
    gruppi: dict[tuple, list[dict]] = defaultdict(list)
    for u in utt.values():
        meta = u.get("meta", {})
        succ = meta.get("success")
        if succ is None:
            continue
        for pid in meta.get("pair_ids") or []:
            gruppi[(u["root"], pid, succ)].append(u)

    titolo("1. RAGGRUPPAMENTO")
    controllo("path unit grezze", 8526, len(gruppi))
    controllo("unita' vincenti", 4263, sum(1 for k in gruppi if k[2] == 1))
    controllo("unita' perdenti", 4263, sum(1 for k in gruppi if k[2] == 0))

    coppie = defaultdict(dict)
    for (root, pid, succ), membri in gruppi.items():
        coppie[(root, pid)][succ] = membri
    complete = sum(1 for d in coppie.values() if set(d) == {0, 1})
    controllo("coppie appaiate complete", 4263, complete)

    # ----------------------------------------------------------------------
    # 2. rimozione delle utterance dell'OP
    # ----------------------------------------------------------------------
    titolo("2. RIMOZIONE DELLE UTTERANCE DELL'OP")

    sfidanti: dict[tuple, list[dict]] = {}
    vuote = 0
    for chiave, membri in gruppi.items():
        autore_op = op.get(chiave[0])
        resto = [u for u in membri if nome_autore(u) != autore_op]
        if not resto:
            vuote += 1
            continue
        sfidanti[chiave] = resto

    tolte = sum(len(v) for v in gruppi.values()) - sum(len(v) for v in sfidanti.values())
    riga("utterance rimosse (autore = OP)", tolte)
    controllo("unita' rimaste vuote", 0, vuote)

    non_singole = sum(1 for v in sfidanti.values()
                      if len({nome_autore(u) for u in v}) != 1)
    controllo("unita' con != 1 sfidante", 0, non_singole)

    # ----------------------------------------------------------------------
    # 3. scarto delle coppie con marcatore residuo o sfidante non valido
    # ----------------------------------------------------------------------
    titolo("3. PULIZIA")

    da_scartare: set[tuple] = set()
    n_marcatore = n_autore_escluso = 0
    for chiave, membri in sfidanti.items():
        if any(DELTA_PATTERN.search(u.get("text") or "") for u in membri):
            da_scartare.add(chiave[:2])
            n_marcatore += 1
        if e_escluso(nome_autore(membri[0])):
            da_scartare.add(chiave[:2])
            n_autore_escluso += 1

    riga("unita' con marcatore di delta residuo", n_marcatore)
    riga("unita' con sfidante bot/[deleted]", n_autore_escluso)
    riga("coppie scartate (gemella inclusa)", len(da_scartare))

    sfidanti = {k: v for k, v in sfidanti.items() if k[:2] not in da_scartare}
    riga("unita' superstiti", len(sfidanti))

    superstiti_coppie = {k[:2] for k in sfidanti}
    spaiate = sum(1 for c in superstiti_coppie
                  if sum(1 for k in sfidanti if k[:2] == c) != 2)
    controllo("coppie spaiate dopo la pulizia", 0, spaiate)

    # ----------------------------------------------------------------------
    # 4. split temporale sui thread
    # ----------------------------------------------------------------------
    titolo("4. FINESTRE TEMPORALI")

    t_thread: dict[str, int] = {}
    for u in utt.values():
        t = u.get("timestamp")
        if t is not None:
            r = u["root"]
            t_thread[r] = min(t_thread.get(r, t), t)

    ordinati = sorted(t_thread, key=t_thread.get)
    taglio = int(len(ordinati) * FRAZIONE_GRAFO)
    grafo_roots = set(ordinati[:taglio])
    data_taglio = pd.to_datetime(t_thread[ordinati[taglio]], unit="s", utc=True)

    riga("thread in W_grafo", len(grafo_roots))
    riga("thread in W_modello", len(ordinati) - taglio)
    riga("data di taglio", str(data_taglio.date()))
    print()
    print("  Le due finestre sono disgiunte: nessuna unita' del modello")
    print("  contribuisce al grafo che ne descrive lo sfidante.")

    # ----------------------------------------------------------------------
    # 5. scrittura delle tabelle
    # ----------------------------------------------------------------------
    titolo("5. SCRITTURA")

    prof = profondita(utt)

    righe = []
    for (root, pid, succ), membri in sorted(sfidanti.items()):
        membri = sorted(membri, key=lambda u: (u.get("timestamp") or 0, u["id"]))
        testi = [DELTA_PATTERN.sub(" ", u.get("text") or "") for u in membri]
        righe.append({
            "unit_id": f"{root}|{pid}|{succ}",
            "pair_id": f"{root}|{pid}",
            "root": root,
            "success": int(succ),
            "sfidante": nome_autore(membri[0]),
            "n_utterance": len(membri),
            "split": W_GRAFO if root in grafo_roots else W_MODELLO,
            "utt_ids": json.dumps([u["id"] for u in membri]),
            "testi": json.dumps(testi),
            "timestamps": json.dumps([u.get("timestamp") for u in membri]),
            "profondita": json.dumps([prof.get(u["id"], 0) for u in membri]),
        })

    units = pd.DataFrame(righe)
    units.to_parquet(DATA / "units.parquet", index=False)

    righe_thread = []
    for cid, c in conv.items():
        righe_thread.append({
            "root": cid,
            "op": c.get("op-userID"),
            "op_title": c.get("op-title") or "",
            "op_text": c.get("op-text-body") or "",
            "timestamp": t_thread.get(cid),
            "split": W_GRAFO if cid in grafo_roots else W_MODELLO,
        })
    threads = pd.DataFrame(righe_thread)
    threads.to_parquet(DATA / "threads.parquet", index=False)

    riga("units.parquet", len(units), "righe")
    riga("threads.parquet", len(threads), "righe")

    # ----------------------------------------------------------------------
    # riepilogo
    # ----------------------------------------------------------------------
    titolo("RIEPILOGO")
    n_coppie = units["pair_id"].nunique()
    riga("coppie appaiate finali", n_coppie)
    riga("unita' finali", len(units))
    bil = units["success"].mean()
    riga("bilanciamento", f"{bil:.1%} / {1 - bil:.1%}", "(atteso 50% / 50%)")

    per_split = units.groupby("split")["success"].agg(["size", "mean"])
    print()
    for split, r in per_split.iterrows():
        riga(f"unita' in W_{split}", int(r["size"]), f"success = {r['mean']:.1%}")
    n_mod = int(per_split.loc[W_MODELLO, "size"])
    riga("coppie utilizzabili per il modello", n_mod // 2)

    med = units.groupby("success")["n_utterance"].mean()
    print()
    riga("utterance/unita' — vincenti", f"{med.get(1, 0):.2f}", "(atteso ~1,42)")
    riga("utterance/unita' — perdenti", f"{med.get(0, 0):.2f}", "(atteso ~1,29)")
    print("\n  ^ confondente residuo dichiarato: entra in B1 come feature,")
    print("    e l'ablation viene riportata anche senza (config B1-).")


if __name__ == "__main__":
    main()
