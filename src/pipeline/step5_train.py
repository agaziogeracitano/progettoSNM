from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, riga, titolo

RIPETIZIONI = 10
FOLD = 5
SEED = 42


def costruisci_coppie(df: pd.DataFrame, colonne: list[str]):
    """Differenze appaiate e simmetrizzate.

    Ritorna X, y, gruppi (root) e gli identificativi di coppia."""
    vin = df[df["success"] == 1].set_index("pair_id")
    per = df[df["success"] == 0].set_index("pair_id")
    comuni = vin.index.intersection(per.index)

    diff = vin.loc[comuni, colonne].to_numpy(float) - per.loc[comuni, colonne].to_numpy(float)
    root = vin.loc[comuni, "root"].to_numpy()

    X = np.vstack([diff, -diff])
    y = np.concatenate([np.ones(len(diff)), np.zeros(len(diff))])
    gruppi = np.concatenate([root, root])
    coppie = np.concatenate([comuni.to_numpy(), comuni.to_numpy()])
    return X, y, gruppi, coppie


def valuta(X, y, gruppi, ripetizioni=RIPETIZIONI, fold=FOLD):
    """GroupKFold ripetuto. Ritorna una riga per fold."""
    rng = np.random.default_rng(SEED)
    gruppi_unici = np.unique(gruppi)
    esiti = []

    for rip in range(ripetizioni):
        perm = rng.permutation(len(gruppi_unici))
        rimappa = {g: i for g, i in zip(gruppi_unici, perm)}
        ordine = np.array([rimappa[g] for g in gruppi])

        for k, (tr, te) in enumerate(GroupKFold(n_splits=fold).split(X, y, ordine)):
            modello = make_pipeline(
                StandardScaler(),
                LogisticRegression(fit_intercept=False, max_iter=2000, C=1.0),
            )
            modello.fit(X[tr], y[tr])
            p = modello.predict_proba(X[te])[:, 1]
            esiti.append({
                "ripetizione": rip,
                "fold": k,
                "accuratezza": float(((p > 0.5) == (y[te] == 1)).mean()),
                "auc": float(roc_auc_score(y[te], p)),
                "n_test": len(te),
            })
    return pd.DataFrame(esiti)


def permutazione(a: np.ndarray, b: np.ndarray, n=20000) -> float:
    """Test di permutazione appaiato sui fold: quanto e' probabile un divario
    grande quanto quello osservato se i due modelli fossero equivalenti."""
    d = a - b
    oss = abs(d.mean())
    rng = np.random.default_rng(SEED)
    segni = rng.choice([-1.0, 1.0], size=(n, len(d)))
    nulla = np.abs((segni * d).mean(axis=1))
    return float(((nulla >= oss).sum() + 1) / (n + 1))


def main() -> None:
    titolo("STEP 5 — ABLATION APPAIATA")

    df = pd.read_parquet(DATA / "features.parquet")
    blocchi = json.loads((DATA / "blocchi.json").read_text(encoding="utf-8"))

    B1 = blocchi["B1"]
    B2 = blocchi["B2"]

    extra: dict[str, list[str]] = {}
    for dichiarazione in sorted(DATA.glob("blocco_*.json")):
        b = json.loads(dichiarazione.read_text(encoding="utf-8"))
        tabella = pd.read_parquet(DATA / b["file"])
        chiave = b.get("chiave", "unit_id")
        prima = len(df)

        if chiave == "unit_id":
            df = df.merge(tabella, on="unit_id", how="left")
            mancanti = int(df[b["colonne"]].isna().any(axis=1).sum())
            if mancanti:
                print(f"  blocco {b['nome']}: {mancanti} unita' senza valore, escluse")
                df = df.dropna(subset=b["colonne"])
        else:
            df = df.merge(tabella.rename(columns={chiave: "sfidante"}),
                          on="sfidante", how="left")
            senza = int(df[b["colonne"]].isna().any(axis=1).sum())
            df[b["colonne"]] = df[b["colonne"]].fillna(0.0)
            print(f"  blocco {b['nome']}: {senza} unita' senza rappresentazione "
                  f"({senza / len(df):.1%}), vettore nullo + flag")

        extra[b["nome"]] = b["colonne"]
        riga(f"blocco {b['nome']} agganciato", len(b["colonne"]),
             f"feature  ({prima} -> {len(df)} unita')")

    # dopo un eventuale dropna le coppie possono essere rimaste spaiate
    completi = df.groupby("pair_id")["success"].transform("size") == 2
    if (~completi).any():
        riga("unita' spaiate rimosse", int((~completi).sum()))
        df = df[completi]

    B3 = extra.get("B3", [])
    B2p = extra.get("B2p", [])   # node2vec (step2b_node2vec.py)
    M = extra.get("M", [])

    B1_meno = [c for c in B1 if c != "n_utterance"]

    config = {
        "B1-": B1_meno,
        "B1": B1,
        "B2": B1 + B2,
    }
    if B2p:
        config["B2'"] = B1 + B2p                 # node2vec al posto di B2
        config["B2+B2'"] = B1 + B2 + B2p         # servono a cose diverse?
    if B3:
        config["B3"] = B1 + B2 + B3
        if B2p:
            config["B3'"] = B1 + B2 + B2p + B3
    if M:
        config["M"] = B1 + B2 + B3 + M

    confronto = {"B1-": None, "B1": "B1-", "B2": "B1", "B2'": "B1",
                 "B2+B2'": "B2'", "B3": "B2", "B3'": "B3", "M": "B3"}

    riga("coppie appaiate", df["pair_id"].nunique())
    riga("configurazioni", len(config) + 1, "(inclusa B0)")
    riga("protocollo", f"{RIPETIZIONI} x GroupKFold-{FOLD}", "raggruppato per root")
    mancanti = [n for n in ("B3", "M") if n not in config]
    if mancanti:
        print(f"\n  non ancora disponibili: {', '.join(mancanti)}")
        print("  (l'ablation gira lo stesso; si riesegue quando sono pronti)")

    # ----------------------------------------------------------------------
    titolo("RISULTATI")
    print(f"  {'config':<8}{'accuratezza':>18}{'AUC':>16}{'vs prec.':>12}{'p':>9}")
    print("  " + "-" * 63)
    print(f"  {'B0':<8}{'50,0%':>18}{'0,500':>16}{'—':>12}{'—':>9}"
          "   riferimento")

    tutti, storico = [], {}
    for nome, colonne in config.items():
        X, y, gruppi, _ = costruisci_coppie(df, colonne)
        res = valuta(X, y, gruppi)
        res["config"] = nome
        res["n_feature"] = len(colonne)
        tutti.append(res)
        storico[nome] = res["accuratezza"]

        acc, sd = res["accuratezza"].mean(), res["accuratezza"].std()
        auc, sda = res["auc"].mean(), res["auc"].std()

        rif = confronto.get(nome)
        if rif is None or rif not in storico:
            delta, p, etichetta = acc - 0.5, "—", "B0"
        else:
            delta = acc - storico[rif].mean()
            p = f"{permutazione(res['accuratezza'].to_numpy(), storico[rif].to_numpy()):.4f}"
            etichetta = rif

        print(f"  {nome:<8}{acc:>11.1%} ± {sd:.1%}{auc:>11.3f} ± {sda:.3f}"
              f"{delta:>+11.1%}{p:>9}   vs {etichetta}")

    risultati = pd.concat(tutti, ignore_index=True)
    risultati.to_parquet(DATA / "risultati.parquet", index=False)

    # ----------------------------------------------------------------------
    titolo("COEFFICIENTI — configurazione completa")

    interpretabili = [n for n in ("B3", "B2", "B1") if n in config]
    nome_finale = interpretabili[0]
    colonne_finali = config[nome_finale]
    X, y, gruppi, _ = costruisci_coppie(df, colonne_finali)
    modello = make_pipeline(
        StandardScaler(),
        LogisticRegression(fit_intercept=False, max_iter=2000, C=1.0),
    ).fit(X, y)
    coef = pd.Series(modello[-1].coef_[0], index=colonne_finali).sort_values(key=abs,
                                                                            ascending=False)
    print(f"  (config {nome_finale}, adattata su tutte le coppie, feature standardizzate)")
    print(f"  Positivo = piu' alto nel vincente.\n")
    for nome, v in coef.head(14).items():
        barra = "#" * int(min(abs(v) * 30, 40))
        print(f"  {nome:<24}{v:>+8.3f}  {barra}")

    # ----------------------------------------------------------------------
    titolo("STRATIFICAZIONE — copertura di rete (RQ1 falsificabile)")

    noti = df.groupby("pair_id")["noto_nel_grafo"].sum()
    print("  Il divario fra le fasce E' la misura del valore dell'informazione")
    print("  di rete: se B2 aiuta, deve aiutare dove la rete e' osservabile.\n")
    print(f"  {'fascia':<28}{'coppie':>9}{'B1':>12}{'B2':>12}{'delta':>10}")
    print("  " + "-" * 71)

    for etichetta, mask in (
        ("entrambi noti", noti == 2),
        ("uno solo noto", noti == 1),
        ("nessuno noto", noti == 0),
    ):
        pid = set(noti[mask].index)
        sub = df[df["pair_id"].isin(pid)]
        if sub["pair_id"].nunique() < 60:
            print(f"  {etichetta:<28}{sub['pair_id'].nunique():>9}   "
                  "(troppo poche, non riportata)")
            continue
        acc = {}
        for nome in ("B1", "B2"):
            Xs, ys, gs, _ = costruisci_coppie(sub, config[nome])
            acc[nome] = valuta(Xs, ys, gs, ripetizioni=4)["accuratezza"].mean()
        print(f"  {etichetta:<28}{sub['pair_id'].nunique():>9}"
              f"{acc['B1']:>11.1%}{acc['B2']:>12.1%}{acc['B2'] - acc['B1']:>+10.1%}")

    print("\n  Se 'entrambi noti' non mostra un guadagno B2 maggiore delle altre")
    print("  fasce, la risposta a RQ1 e' negativa — ed e' un risultato.")


if __name__ == "__main__":
    main()
