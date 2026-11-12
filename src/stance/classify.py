from __future__ import annotations

import argparse
import json
import time
from collections import Counter, deque

import pandas as pd
import requests

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    DATA, W_MODELLO, chiave_groq, modello_groq, riga, titolo,
)

URL = "https://api.groq.com/openai/v1/chat/completions"
USCITA = DATA / "stance.jsonl"

ETICHETTE = ("CONCESSIVE", "DIRECT", "CLARIFYING", "HOSTILE", "AGREE", "OTHER")

FALLITO = "FALLITO"

BATCH = 10
TENTATIVI = 4               # autocorrezioni del formato
TENTATIVI_RETE = 10         # ritentativi su rete/429/5xx, con backoff
CAP_CARATTERI = 2000        # 29% delle unita' viene troncato
BUDGET_TPM = 8000           # limite del free tier, letto dagli header
MARGINE = 0.80              # si usa l'80% del budget per non sfiorare il muro

SYSTEM = f"""You label HOW a challenger disagrees with an opinion post on Reddit's r/ChangeMyView.

The challenger is arguing against the original poster (OP). They almost always
disagree - your job is to classify the MANNER, not whether they disagree.

{ETICHETTE[0]}  - grants something valid in the OP's view before objecting
{ETICHETTE[1]}      - contradicts head-on, no concession offered
{ETICHETTE[2]}  - asks for clarification or redefines the terms rather than opposing
{ETICHETTE[3]}     - disagrees with a contemptuous, mocking or hostile tone toward the OP
{ETICHETTE[4]}       - actually agrees with the OP
{ETICHETTE[5]}       - none of the above

Rules:
- Judge tone toward the OP, not toward third parties.
- A polite opening does not make it CONCESSIVE; there must be a real concession.
- If both concessive and hostile elements appear, pick the dominant one.

Reply with ONLY a JSON object {{"labels": [...]}} containing exactly N strings,
in the same order as the numbered items. No commentary."""


def accorcia(testo: str, cap: int = CAP_CARATTERI) -> str:
    """Testa e coda: la mossa concessiva sta di solito in apertura, il tono in
    chiusura. Tagliare solo la fine perderebbe il secondo."""
    testo = " ".join(testo.split())
    if len(testo) <= cap:
        return testo
    return testo[: int(cap * 0.7)] + " [...] " + testo[-int(cap * 0.3):]


def _durata(v: str | None) -> float:
    """Converte '1m13.5s' / '112ms' / '4.16s' in secondi."""
    if not v:
        return 0.0
    v = v.strip()
    try:
        if v.endswith("ms"):
            return float(v[:-2]) / 1000.0
        tot, num = 0.0, ""
        for c in v:
            if c.isdigit() or c == ".":
                num += c
            elif c == "m":
                tot += float(num or 0) * 60
                num = ""
            elif c == "s":
                tot += float(num or 0)
                num = ""
        return tot + (float(num) if num else 0.0)
    except ValueError:
        return 0.0


class Limitatore:
    """Governa il ritmo delle chiamate.

    Groq impone DUE limiti diversi, entrambi vincolanti:
      - token al minuto (8.000), che si ricarica in fretta
      - richieste al GIORNO (1.000), che si ricarica lentamente

    La prima versione stimava i token da se' (lunghezza/4). Quando
    sottostimava scattava un 429, che costa una richiesta del budget
    giornaliero E un'attesa imposta anche di 19 minuti. Ora la stima serve
    solo prima della primissima chiamata: da li' in poi si usano gli header
    che il server restituisce, che sono la verita'.
    """

    def __init__(self, budget: int = BUDGET_TPM):
        self.budget = budget * MARGINE
        self.storia: deque = deque()
        self.token_residui: float | None = None
        self.ricarica_token: float = 0.0
        self.richieste_residue: int | None = None

    def _pulisci(self) -> float:
        ora = time.time()
        while self.storia and ora - self.storia[0][0] > 60:
            self.storia.popleft()
        return sum(t for _, t in self.storia)

    def attendi(self, previsti: int) -> None:
        # 1. quello che dice il server ha la precedenza sulla mia stima
        if self.token_residui is not None and self.token_residui < previsti:
            pausa = max(self.ricarica_token, 1.0) + 1
            print(f"    token residui {self.token_residui:.0f} < {previsti} "
                  f"richiesti, pausa {pausa:.0f}s", flush=True)
            time.sleep(pausa)
            self.token_residui = None      # dopo la ricarica non so piu': rileggo
            return

        # 2. rete di sicurezza sulla finestra locale, per la prima chiamata
        while True:
            usati = self._pulisci()
            if usati + previsti <= self.budget or not self.storia:
                return
            pausa = 61 - (time.time() - self.storia[0][0])
            if pausa <= 0:
                return
            print(f"    budget locale esaurito, pausa {pausa:.0f}s", flush=True)
            time.sleep(pausa)

    def aggiorna(self, headers) -> None:
        """Legge i contatori autorevoli dalla risposta del server."""
        try:
            t = headers.get("x-ratelimit-remaining-tokens")
            if t is not None:
                self.token_residui = float(t)
            self.ricarica_token = _durata(headers.get("x-ratelimit-reset-tokens"))
            r = headers.get("x-ratelimit-remaining-requests")
            if r is not None:
                self.richieste_residue = int(float(r))
        except (TypeError, ValueError):
            pass

    def registra(self, token: int) -> None:
        self.storia.append((time.time(), token))


class ReteIrraggiungibile(RuntimeError):
    """La rete non torna dopo tutti i ritentativi. Si interrompe la corsa
    lasciando intatto il salvataggio: si riprende rilanciando lo script."""


def posta(sessione, chiave, modello, messaggi, limitatore, previsti):
    """Una POST, con ritentativi su errori di rete, 429 e 5xx.

    Separata dalla validazione del formato: un blip di rete non deve consumare
    un tentativo di autocorrezione del modello, e non deve uccidere una corsa
    da diverse ore (successo gia' visto sul campo: un getaddrinfo fallito).
    """
    attesa = 5.0
    for _ in range(TENTATIVI_RETE):
        limitatore.attendi(previsti)
        try:
            r = sessione.post(
                URL,
                headers={"Authorization": f"Bearer {chiave}"},
                json={"model": modello, "temperature": 0, "max_tokens": 800,
                      "response_format": {"type": "json_object"},
                      "messages": messaggi},
                timeout=180,
            )
        except requests.RequestException as e:
            print(f"    rete: {type(e).__name__}, riprovo fra {attesa:.0f}s",
                  flush=True)
            time.sleep(attesa)
            attesa = min(attesa * 2, 300)
            continue

        if r.status_code == 429:
            limitatore.aggiorna(r.headers)
            pausa = float(r.headers.get("retry-after", 20))
            print(f"    429 dal server, pausa {pausa:.0f}s "
                  f"(richieste residue oggi: {limitatore.richieste_residue})",
                  flush=True)
            time.sleep(pausa + 1)
            continue
        if r.status_code >= 500:
            print(f"    HTTP {r.status_code} dal server, riprovo fra {attesa:.0f}s",
                  flush=True)
            time.sleep(attesa)
            attesa = min(attesa * 2, 300)
            continue
        if not r.ok:
            print(f"    HTTP {r.status_code}: {r.text[:150]}", flush=True)
            time.sleep(5)
            continue

        limitatore.aggiorna(r.headers)
        return r

    raise ReteIrraggiungibile(
        f"nessuna risposta utile dopo {TENTATIVI_RETE} tentativi")


def chiama(sessione, chiave, modello, testi, limitatore) -> list[str]:
    """Una chiamata su un batch, con validazione e autocorrezione."""
    numerati = "\n\n".join(f"{i + 1}. {t}" for i, t in enumerate(testi))
    messaggi = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"N={len(testi)}\n\n{numerati}"},
    ]
    previsti = len(numerati) // 4 + 400

    for tentativo in range(TENTATIVI):
        r = posta(sessione, chiave, modello, messaggi, limitatore, previsti)
        d = r.json()
        limitatore.registra(d.get("usage", {}).get("total_tokens", previsti))
        grezzo = d["choices"][0]["message"]["content"]

        # validazione: JSON, lunghezza dell'array, etichette ammesse
        errore = None
        try:
            dati = json.loads(grezzo)
            et = dati.get("labels") if isinstance(dati, dict) else dati
            if not isinstance(et, list):
                errore = "the JSON must contain a list under 'labels'"
            elif len(et) != len(testi):
                errore = f"you returned {len(et)} labels, exactly {len(testi)} are required"
            else:
                et = [str(x).strip().upper() for x in et]
                fuori = [x for x in et if x not in ETICHETTE]
                if fuori:
                    errore = f"invalid labels {fuori[:3]}; allowed: {list(ETICHETTE)}"
                else:
                    return et
        except json.JSONDecodeError:
            errore = "the reply was not valid JSON"

        # autocorrezione: l'errore torna al modello invece di riprovare uguale
        print(f"    tentativo {tentativo + 1}: {errore}", flush=True)
        messaggi += [
            {"role": "assistant", "content": grezzo[:400]},
            {"role": "user", "content":
             f"That was wrong: {errore}. Reply again with ONLY "
             f'{{"labels": [...]}} containing exactly {len(testi)} labels.'},
        ]

    return [FALLITO] * len(testi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None,
                    help="classifica solo le prime N unita' (prova)")
    args = ap.parse_args()

    titolo("STEP 4 — STANCE DETECTION")

    chiave, modello = chiave_groq(), modello_groq()
    units = pd.read_parquet(DATA / "units.parquet")
    units = units[units["split"] == W_MODELLO]

    # ripresa: si rifanno solo le unita' non ancora salvate
    fatte: dict[str, str] = {}
    if USCITA.exists():
        for r in USCITA.read_text(encoding="utf-8").splitlines():
            if r.strip():
                d = json.loads(r)
                fatte[d["unit_id"]] = d["stance"]

    da_fare = units[~units["unit_id"].isin(fatte)]
    if args.limite:
        da_fare = da_fare.head(args.limite)

    riga("modello", modello)
    riga("unita' totali (W_modello)", len(units))
    riga("gia' classificate", len(fatte))
    riga("da classificare", len(da_fare))
    riga("chiamate previste", (len(da_fare) + BATCH - 1) // BATCH)

    if da_fare.empty:
        print("\n  niente da fare.")
    else:
        sessione = requests.Session()
        limitatore = Limitatore()
        t0 = time.time()

        with open(USCITA, "a", encoding="utf-8") as out:
            for i in range(0, len(da_fare), BATCH):
                blocco = da_fare.iloc[i:i + BATCH]
                testi = [accorcia(" ".join(json.loads(t)))
                         for t in blocco["testi"]]
                try:
                    etichette = chiama(sessione, chiave, modello, testi, limitatore)
                except (ReteIrraggiungibile, KeyboardInterrupt) as e:
                    print(f"\n  INTERROTTO: {e}")
                    print(f"  salvate {len(fatte) + i} unita'. Nulla e' andato perso:")
                    print("  rilancia lo stesso comando per riprendere da qui.\n")
                    break

                for uid, et in zip(blocco["unit_id"], etichette):
                    out.write(json.dumps({"unit_id": uid, "stance": et}) + "\n")
                out.flush()          # salvataggio incrementale reale

                res = limitatore.richieste_residue
                if res is not None:
                    servono = (len(da_fare) - i - len(blocco) + BATCH - 1) // BATCH
                    if res < servono:
                        print(f"\n  ATTENZIONE: restano {res} richieste oggi ma ne"
                              f" servono ancora {servono}.")
                        print("  La corsa si fermera' e riprendera' domani: "
                              "rilancia lo stesso comando.\n", flush=True)

                fatti = i + len(blocco)
                trascorso = time.time() - t0
                stimato = trascorso / fatti * (len(da_fare) - fatti)
                print(f"  {fatti:>5}/{len(da_fare)}  "
                      f"({fatti / len(da_fare):5.1%})  "
                      f"restano ~{stimato / 60:.0f} min", flush=True)

    # ----------------------------------------------------------------------
    titolo("ESITO")

    righe = [json.loads(r) for r in USCITA.read_text(encoding="utf-8").splitlines()
             if r.strip()]
    st = pd.DataFrame(righe).drop_duplicates("unit_id")
    st.to_parquet(DATA / "stance.parquet", index=False)

    conta = Counter(st["stance"])
    for et in ETICHETTE:
        n = conta.get(et, 0)
        riga(et, n, f"({n / len(st):.1%})" if len(st) else "")

    n_falliti = conta.get(FALLITO, 0)
    if n_falliti:
        riga(FALLITO, n_falliti, f"({n_falliti / len(st):.1%})  <- escluse dal blocco M")

    dominante = conta.most_common(1)[0]
    if dominante[1] / len(st) > 0.75:
        print(f"\n  ATTENZIONE: {dominante[0]} copre il {dominante[1] / len(st):.0%}.")
        print("  Con una distribuzione cosi' squilibrata il blocco M non puo'")
        print("  separare le due unita' di una coppia. Da verificare col kappa.")

    riga("\nstance.parquet", len(st), "righe")
    print("\n  PROSSIMO PASSO OBBLIGATORIO: src/stance/validate.py")
    print("  Finche' il kappa contro l'annotazione umana non supera 0,4,")
    print("  questo blocco non entra nel modello.")


if __name__ == "__main__":
    main()
