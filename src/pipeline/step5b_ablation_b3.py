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

# le 5 misure contate a mano su insiemi di parole (step3b_semantic.py)
B3_LEX = ["ip_contenuto_coperto", "ip_jaccard", "ip_stop_ripreso",
          "ip_parole_nuove", "diversita_lessicale"]
# le 3 che passano da Sentence-BERT
B3_SEM = ["sim_semantica_op", "auto_similarita", "ha_auto_similarita"]
B3_SEM_PURO = ["sim_semantica_op", "auto_similarita"]


def costruisci_coppie(df: pd.DataFrame, colonne: list[str]):
    """Differenze appaiate e simmetrizzate (identica a step5_train.py)."""
    vin = df[df["success"] == 1].set_index("pair_id")
    per = df[df["success"] == 0].set_index("pair_id")
    comuni = vin.index.intersection(per.index)

    diff = (vin.loc[comuni, colonne].to_numpy(float)
            - per.loc[comuni, colonne].to_numpy(float))
    root = vin.loc[comuni, "root"].to_numpy()

    X = np.vstack([diff, -diff])
    y = np.concatenate([np.ones(len(diff)), np.zeros(len(diff))])
    gruppi = np.concatenate([root, root])
    return X, y, gruppi


def valuta(X, y, gruppi, ripetizioni=RIPETIZIONI, fold=FOLD):
    """GroupKFold ripetuto. Ritorna una riga per fold."""
    rng = np.random.default_rng(SEED)
    gruppi_unici = np.unique(gruppi)
    esiti = []

    for rip in range(ripetizioni):
        # GroupKFold non mescola: rimescolo io l'ordine dei gruppi
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
            })
    return pd.DataFrame(esiti)


def permutazione(a: np.ndarray, b: np.ndarray, n=20000) -> float:
    """Test di permutazione appaiato a inversione di segno."""
    d = a - b
    oss = abs(d.mean())
    rng = np.random.default_rng(SEED)
    segni = rng.choice([-1.0, 1.0], size=(n, len(d)))
    nulla = np.abs((segni * d).mean(axis=1))
    return float(((nulla >= oss).sum() + 1) / (n + 1))


def esegui(df, config: dict[str, list[str]], nome_riferimento: str):
    """Valuta ogni configurazione e la confronta col riferimento indicato."""
    tabelle, sintesi = [], []
    rif_acc = None

    for nome, colonne in config.items():
        X, y, gruppi = costruisci_coppie(df, colonne)
        res = valuta(X, y, gruppi)
        res["config"] = nome
        res["n_feature"] = len(colonne)
        tabelle.append(res)

        acc = res["accuratezza"].to_numpy()
        if nome == nome_riferimento:
            rif_acc = acc

        voce = {
            "config": nome,
            "n_feature": len(colonne),
            "accuratezza": float(acc.mean()),
            "accuratezza_sd": float(acc.std()),
            "auc": float(res["auc"].mean()),
            "auc_sd": float(res["auc"].std()),
            "riferimento": None if nome == nome_riferimento else nome_riferimento,
        }
        if nome != nome_riferimento:
            voce["delta"] = float(acc.mean() - rif_acc.mean())
            voce["p"] = permutazione(acc, rif_acc)
        sintesi.append(voce)

        d = "" if nome == nome_riferimento else f"{voce['delta']:+.1%}"
        p = "" if nome == nome_riferimento else f"{voce['p']:.4f}"
        print(f"  {nome:<26}{acc.mean():>8.1%} ± {acc.std():.1%}"
              f"{res['auc'].mean():>10.3f} ± {res['auc'].std():.3f}"
              f"{d:>10}{p:>9}")

    return pd.concat(tabelle, ignore_index=True), sintesi


def main() -> None:
    titolo("STEP 5b — ABLATION SEPARATA DEI SOTTOBLOCCHI DI B3")

    blocchi = json.loads((DATA / "blocchi.json").read_text(encoding="utf-8"))
    B1, B2 = blocchi["B1"], blocchi["B2"]
    b3 = json.loads((DATA / "blocco_B3.json").read_text(encoding="utf-8"))
    B3 = b3["colonne"]

    assert sorted(B3) == sorted(B3_LEX + B3_SEM), (
        "le colonne di blocco_B3.json non coincidono con la partizione "
        "lessicale/semantica dichiarata in questo script"
    )

    df = pd.read_parquet(DATA / "features.parquet")
    tab = pd.read_parquet(DATA / b3["file"])
    df = df.merge(tab, on="unit_id", how="left")
    mancanti = int(df[B3].isna().any(axis=1).sum())
    if mancanti:
        riga("unita' senza valori B3, escluse", mancanti)
        df = df.dropna(subset=B3)

    completi = df.groupby("pair_id")["success"].transform("size") == 2
    if (~completi).any():
        riga("unita' spaiate rimosse", int((~completi).sum()))
        df = df[completi]

    riga("coppie appaiate", df["pair_id"].nunique())
    riga("protocollo", f"{RIPETIZIONI} x GroupKFold-{FOLD}", "raggruppato per root")

    # ----------------------------------------------------------------------
    titolo("1. QUANTO VALE CIASCUN SOTTOBLOCCO DA SOLO (rif. B1+B2)")
    print(f"  {'config':<26}{'accuratezza':>14}{'AUC':>16}{'vs rif.':>10}{'p':>9}")
    print("  " + "-" * 75)

    tab1, sintesi1 = esegui(df, {
        "B1+B2": B1 + B2,
        "  +B3lex (5)": B1 + B2 + B3_LEX,
        "  +B3sem (3)": B1 + B2 + B3_SEM,
        "  +B3sem senza flag (2)": B1 + B2 + B3_SEM_PURO,
        "  +B3 completo (8)": B1 + B2 + B3,
    }, "B1+B2")

    print("\n  Il blocco lessicale da solo recupera l'intero guadagno di B3.")

    # ----------------------------------------------------------------------
    titolo("2. IL TEST DECISIVO: LA SEMANTICA AGGIUNGE QUALCOSA AL LESSICO?")
    print(f"  {'config':<26}{'accuratezza':>14}{'AUC':>16}{'vs rif.':>10}{'p':>9}")
    print("  " + "-" * 75)

    tab2, sintesi2 = esegui(df, {
        "B1+B2+B3lex": B1 + B2 + B3_LEX,
        "  +sem senza flag (2)": B1 + B2 + B3_LEX + B3_SEM_PURO,
        "  +B3 completo (8)": B1 + B2 + B3,
    }, "B1+B2+B3lex")

    print("\n  Con il lessico gia' nel modello, la semantica non aggiunge nulla:")
    print("  il suo segnale e' interamente RIDONDANTE, non assente.")

    # ----------------------------------------------------------------------
    titolo("3. COEFFICIENTI STANDARDIZZATI — modello completo B1+B2+B3")

    colonne_finali = B1 + B2 + B3
    X, y, gruppi = costruisci_coppie(df, colonne_finali)
    modello = make_pipeline(
        StandardScaler(),
        LogisticRegression(fit_intercept=False, max_iter=2000, C=1.0),
    ).fit(X, y)
    coef = pd.Series(modello[-1].coef_[0], index=colonne_finali)

    print("  (adattato su tutte le coppie. Positivo = piu' alto nel vincente)\n")
    for nome in B3:
        marca = "LEX" if nome in B3_LEX else "SEM"
        v = coef[nome]
        print(f"  [{marca}] {nome:<24}{v:>+8.3f}  {'#' * int(min(abs(v) * 40, 40))}")

    # ----------------------------------------------------------------------
    titolo("SCRITTURA")

    risultati = pd.concat(
        [tab1.assign(confronto="vs B1+B2"), tab2.assign(confronto="vs B1+B2+B3lex")],
        ignore_index=True,
    )
    risultati.to_parquet(DATA / "risultati_b3.parquet", index=False)
    riga("risultati_b3.parquet", len(risultati), "righe (50 per configurazione)")

    (DATA / "ablation_b3.json").write_text(json.dumps({
        "protocollo": {
            "ripetizioni": RIPETIZIONI, "fold": FOLD, "seed": SEED,
            "permutazioni": 20000,
            "coppie": int(df["pair_id"].nunique()),
        },
        "sottoblocchi": {
            "B3lex": B3_LEX, "B3sem": B3_SEM, "B3sem_puro": B3_SEM_PURO,
        },
        "confronto_1_vs_B1B2": sintesi1,
        "confronto_2_vs_B1B2B3lex": sintesi2,
        "coefficienti_B3": {n: float(coef[n]) for n in B3},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    riga("ablation_b3.json", "riepilogo", "medie, delta, p-value, coefficienti")


if __name__ == "__main__":
    main()
