from __future__ import annotations

import json
import os
import re
from pathlib import Path

# --------------------------------------------------------------------------
# percorsi
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw"
CORPUS = RAW / "corpus2"           # Winning Arguments Corpus (dataset principale)
UTTERANCES = CORPUS / "utterances.jsonl"
CONVERSATIONS = CORPUS / "conversations.json"

CORPUS_CGA = RAW / "corpus"        # CGA-CMV — non usato dalla pipeline

DATA = ROOT / "data"               # output della pipeline
DATA.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# finestre temporali
# --------------------------------------------------------------------------
FRAZIONE_GRAFO = 0.50

W_GRAFO = "grafo"
W_MODELLO = "modello"


# --------------------------------------------------------------------------
# variabili d'ambiente (.env)
# --------------------------------------------------------------------------
ENV_FILE = ROOT / ".env"


def carica_env(path: Path = ENV_FILE) -> dict[str, str]:
    """Legge il .env senza dipendere da python-dotenv.

    Le variabili gia' presenti nell'ambiente hanno la precedenza, cosi' si puo'
    sovrascrivere al volo dalla riga di comando senza toccare il file.
    """
    valori: dict[str, str] = {}
    if path.exists():
        for r in path.read_text(encoding="utf-8").splitlines():
            r = r.strip()
            if not r or r.startswith("#") or "=" not in r:
                continue
            k, _, v = r.partition("=")
            valori[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in valori.items():
        os.environ.setdefault(k, v)
    return valori


def chiave_groq() -> str:
    """Restituisce la API key, con un messaggio utile se manca."""
    carica_env()
    k = os.environ.get("GROQ_API_KEY", "").strip()
    if not k:
        raise SystemExit(
            "\nGROQ_API_KEY non impostata.\n\n"
            f"  1. crea una chiave gratuita su https://console.groq.com/keys\n"
            f"  2. incollala in {ENV_FILE}  ->  GROQ_API_KEY=gsk_...\n\n"
            "  (il file e' gia' in .gitignore: la chiave non finisce su git)\n"
        )
    return k


def modello_groq() -> str:
    carica_env()
    return os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")


# --------------------------------------------------------------------------
# filtri
# --------------------------------------------------------------------------

# account non attribuibili a una persona
ESCLUSI = frozenset({
    "[deleted]", "[removed]", "", None,
    "DeltaBot", "A_Mirror", "AutoModerator", "Snapshillbot",
})

# marcatore di delta, da rimuovere dal testo
DELTA_PATTERN = re.compile(r"!delta|&#8710;|&#916;|∆|Δ|&amp;#8710;")

URL_PATTERN = re.compile(r"https?://\S+")


def e_escluso(nome) -> bool:
    return nome in ESCLUSI


def nome_autore(u: dict):
    """Nel Winning Arguments Corpus il campo e' 'user'; in altri corpora
    ConvoKit e' 'speaker'. Puo' essere una stringa o un oggetto con dentro
    l'id."""
    sp = u.get("user") or u.get("speaker")
    return sp.get("id") if isinstance(sp, dict) else sp


def campo_padre(u: dict):
    """Nel file e' 'reply-to' con il TRATTINO; la documentazione dice
    'reply_to'. Costata mezza giornata."""
    return u.get("reply-to") or u.get("reply_to")


# --------------------------------------------------------------------------
# lettura del corpus
# --------------------------------------------------------------------------

def itera_utterance(path: Path = UTTERANCES):
    """Scorre il .jsonl una riga alla volta, senza caricarlo tutto."""
    with open(path, encoding="utf-8") as f:
        for riga in f:
            riga = riga.strip()
            if riga:
                yield json.loads(riga)


def carica_utterance(path: Path = UTTERANCES) -> dict[str, dict]:
    """Carica tutto il corpus in memoria come {id: utterance}.

    ~350 MB su disco, circa 1.5 GB in RAM. Serve perche' quasi ogni step ha
    bisogno di risalire dal figlio al padre.
    """
    return {u["id"]: u for u in itera_utterance(path)}


def carica_conversazioni(path: Path = CONVERSATIONS) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mappa_op(conversazioni: dict[str, dict] | None = None) -> dict[str, str]:
    """{root -> userID dell'OP}. Indispensabile: senza sapere chi e' l'OP non
    si possono escludere le sue utterance, e senza quell'esclusione il modello
    impara un artefatto (l'OP compare nel 100% dei percorsi vincenti)."""
    if conversazioni is None:
        conversazioni = carica_conversazioni()
    return {cid: c.get("op-userID") for cid, c in conversazioni.items()}


# --------------------------------------------------------------------------
# utilita' di stampa
# --------------------------------------------------------------------------

def titolo(testo: str) -> None:
    print()
    print("=" * 64)
    print(testo)
    print("=" * 64)


def riga(etichetta: str, valore, nota: str = "") -> None:
    v = f"{valore:,}".replace(",", ".") if isinstance(valore, int) else valore
    print(f"  {etichetta:<44}{v:>12}  {nota}")


def controllo(etichetta: str, atteso, ottenuto) -> bool:
    """Verifica un conteggio atteso e lo stampa. Ogni step ne ha qualcuno:
    servono ad accorgersi subito se il corpus o la logica cambiano sotto i
    piedi."""
    ok = atteso == ottenuto
    stato = "OK" if ok else f"ATTESO {atteso}"
    print(f"  [{'v' if ok else 'X'}] {etichetta:<40}{ottenuto:>10}   {stato}")
    return ok
