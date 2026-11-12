# Topologia vs Lessico: Dinamiche di Persuasione su r/ChangeMyView

Analisi ablativa dei predittori di successo persuasivo su
[r/ChangeMyView](https://www.reddit.com/r/changemyview/), condotta sul Winning
Arguments Corpus (Tan et al., 2016).

Corso di *Analisi di Social Network e Media* — Università della Calabria,
Corso di Laurea Magistrale in Ingegneria Informatica, A.A. 2025-2026.
**Agazio Geracitano** (269290) · **Vittorio Iusi** (269170)

La relazione completa è in [`Relazione progetto SNM.pdf`](Relazione%20progetto%20SNM.pdf).

---

## La domanda

Su r/ChangeMyView un utente (l'**OP**) espone un'opinione dichiarandosi disposto
a cambiarla; altri utenti — gli **sfidanti** — provano a convincerlo, e chi ci
riesce riceve un **delta**.

Abbiamo chiesto: *a parità di argomento, di OP e di contesto, cosa distingue lo
sfidante che persuade da quello che fallisce — la sua **posizione nella rete**,
il **contenuto** di ciò che scrive, o il **tema** di cui parla abitualmente?*

### Il disegno appaiato

Il corpus fornisce **coppie appaiate**: nella stessa discussione, due sfidanti
argomentano con lo stesso OP e uno solo dei due ottiene il delta. Il compito che
abbiamo impostato non è quindi «questo commento vince?» ma **«fra questi due,
chi ha vinto?»**.

La scelta è metodologica, non pratica. Alcune discussioni sono intrinsecamente
più facili di altre, ma sono **confondenti** che dipendono dal thread e non dallo
sfidante: confrontando due sfidanti *della stessa discussione* si annullano per
costruzione, invece che essere controllati a posteriori.

Il classificatore predice chi dei due ha vinto, ma **quella predizione non è
l'obiettivo: è lo strumento di misura**. Ciò che misuriamo sono le differenze di
accuratezza fra configurazioni consecutive dell'ablazione.

## Le quattro domande di ricerca

| | |
|---|---|
| **RQ1** | La **posizione topologica** dello sfidante nella rete di interazione — centralità, ruolo di ponte fra comunità, reputazione persuasiva pregressa — predice l'esito del confronto? |
| **RQ2** | Il risultato di RQ1 dipende dalla scelta arbitraria delle feature topologiche? Una rappresentazione **appresa** dalla rete (node2vec) fa meglio? |
| **RQ3** | Il **contenuto** dell'intervento contribuisce alla predizione? E il contributo è di natura **lessicale** (sovrapposizione di parole con l'OP) o **semantica** (prossimità di significato)? |
| **RQ4** | Abbandonando il compito predittivo: nell'**influence maximization**, lo schema con cui si assegnano le probabilità di attivazione agli archi è una scelta neutra? |

L'ordine è deliberato. RQ1 e RQ2 esauriscono la via strutturale — prima con
misure scritte a mano, poi con una rappresentazione appresa — e solo dopo si
passa al contenuto. RQ4 cambia compito: non predice più nulla, ma torna sulla
rete con una domanda di ottimizzazione.

---

## Che cosa abbiamo fatto

### 1. Preprocessing: dal corpus grezzo alle coppie pulite

Il lavoro di preparazione è la parte più delicata del progetto, perché il corpus
contiene tracce dirette del processo di etichettatura.

- **Ricostruzione delle *path unit*.** L'etichetta `meta.success` non è per
  commento ma **per percorso**, replicata su tutta la catena fino all'eventuale
  delta. Abbiamo raggruppato i commenti per la terna *(discussione, coppia,
  esito)*, così che ogni unità di analisi sia «uno sfidante, una discussione, un
  esito» e nessuna osservazione venga contata più volte.
- **Rimozione del label leakage.** Il percorso del corpus comprende anche i turni
  dell'OP, incluso quello che *contiene il delta*. Abbiamo misurato l'entità del
  problema prima di intervenire, e rimosso tutte le utterance dell'OP **prima**
  del calcolo di qualsiasi feature.
- **Pulizia simmetrica.** Scartati i percorsi con marcatori di delta residui nel
  testo dello sfidante e quelli di account automatici o cancellati (`[deleted]`
  non è un utente: è un segnaposto che collasserebbe centinaia di persone in un
  unico nodo della rete). Lo scarto è **simmetrico**: eliminare un'unità elimina
  anche la sua gemella, perché una coppia incompleta non ammette confronto.
- **Due finestre temporali disgiunte.** Ordinando le discussioni per timestamp e
  tagliando a metà, abbiamo separato la finestra che costruisce la rete e le
  statistiche storiche da quella che addestra e valuta il modello. Senza questa
  separazione la centralità di un'unità includerebbe le interazioni che si vuole
  predire e la reputazione includerebbe il delta da indovinare: abbiamo misurato
  di quanto l'auto-inclusione gonfia il segnale, e scelto di eliminare il
  problema *per costruzione* anziché correggerlo a posteriori — dichiarandone il
  costo in coppie inutilizzabili.

### 2. I blocchi di feature

| Blocco | Che cos'è |
|---|---|
| **B1** — superficiali (15 col.) | Quantità (numero di commenti, parole totali/medie/del più lungo), fonti e struttura (URL, citazioni, paragrafi, elenchi, punti interrogativi, connettivi argomentativi), posizione nell'albero del thread, tempo. Non richiedono di sapere *chi* sia l'autore. È la baseline che tiene onesto il progetto. |
| **B2** — topologiche (13 col.) | Grado e forza, clustering locale, PageRank pesato, betweenness approssimata per campionamento di pivot, **coefficiente di partecipazione di Guimerà–Amaral** sulla partizione di Louvain, numero di comunità nel vicinato, reputazione pregressa con smoothing di Laplace, indicatore di copertura. |
| **B2′** — node2vec (64 col.) | Rappresentazione appresa in modo non supervisionato da camminate distorte sul grafo. È un'**alternativa** a B2, non un gradino aggiuntivo della scala. |
| **B3** — contenuto (8 col.) | Cinque misure di **interplay lessicale** con il post dell'OP (copertura, Jaccard, novità, rispecchiamento delle parole funzionali, diversità type/token) e tre **semantiche** con Sentence-BERT (similarità con l'OP, auto-similarità fra repliche successive, indicatore di disponibilità). |

Il **grafo di interazione** su cui poggia B2 è costruito a parte: grafo non
orientato e pesato degli utenti attivi, un arco per ogni relazione di risposta
della finestra, autorisposte e bot esclusi, direzione deliberatamente ignorata.

Alcune scelte che abbiamo dovuto motivare: `meta.score` (gli upvote) è escluso
da B1 perché osservato *dopo* il commento ed endogeno al delta; il coefficiente
di partecipazione non esiste in `networkx` ed è implementato direttamente sulla
partizione di Louvain; la frazione complementare alla novità lessicale è omessa
perché introdurrebbe collinearità perfetta; gli sfidanti assenti dalla rete
ricevono valori strutturali imputati a zero (posizione *indefinita*, non bassa) e
un tasso di delta neutro, affiancati da un indicatore che permette al modello di
distinguere lo zero osservato dal dato mancante.

### 3. Il protocollo sperimentale

- **Differenza appaiata.** Ogni coppia è compressa in una riga sola fatta di
  differenze fra le feature dei due sfidanti: tutto ciò che hanno in comune si
  annulla per sottrazione. È l'impianto della regressione logistica condizionale
  (modello di Bradley–Terry lineare), **senza intercetta**, perché il modello
  deve soddisfare parità e coerenza — due sfidanti identici devono dare 0,5.
- **Simmetrizzazione.** Ogni coppia entra due volte, a ruoli scambiati: le classi
  risultano bilanciate per costruzione e l'ordine cessa di veicolare
  informazione, chiudendo la scorciatoia «predici sempre 1».
- **Validazione.** GroupKFold ripetuto, con i fold costruiti su **discussioni
  intere** perché il modello non veda in addestramento il gemello dell'istanza di
  test, e l'ordine dei thread permutato a ogni ripetizione — `GroupKFold` è
  deterministico e da solo darebbe sempre la stessa partizione.
- **Test di permutazione appaiato.** I fold delle configurazioni a confronto sono
  generati con lo stesso seme, quindi testano sulle stesse discussioni: si
  permutano i segni delle differenze fold-per-fold invece di mescolare i due
  campioni.
- **Ablazione estensibile.** Lo script di addestramento non conosce in anticipo
  quali blocchi esistono: scandisce i descrittori prodotti dagli altri script e
  aggancia quello che trova, così un nuovo blocco di feature compare da solo
  nella tabella senza toccare il codice di addestramento.

### 4. Influence maximization (RQ4)

Cambio di compito: individuare l'insieme di *k* utenti la cui attivazione
massimizza la diffusione attesa nella rete.

- **Scelta del grafo.** Il grafo dei delta, candidato naturale, è aciclico e con
  grado uscente medio sotto 1 — ogni cascata si estinguerebbe al primo passo.
  Abbiamo quindi lavorato sulla rete di risposta, usando i delta solo per
  *stimare* le probabilità sugli archi.
- **Due modelli di diffusione in parallelo**, Independent Cascade e Linear
  Threshold, per verificare che le conclusioni non dipendano dal modello scelto.
- **Tre schemi di probabilità**: uniforme, weighted cascade (1/k del ricevente) e
  stimato dai dati con smoothing bayesiano verso la media globale. I valori
  stimati sono riscalati sulla media dello schema uniforme, così il confronto
  isola la *forma* dell'assegnazione e non la sua scala assoluta, che un campione
  di thread selezionati non identifica.
- **Quattro algoritmi**: Greedy, CELF, RIS e le euristiche di riferimento
  (top-degree, PageRank), come metro di paragone — se l'ottimizzazione con
  garanzia $(1-1/e)$ non batte un'euristica elementare, il suo costo non è
  giustificato.
- **Un difetto diagnosticato e corretto.** In una prima implementazione CELF
  risultava inferiore all'euristica per grado: un esito *incompatibile con la
  garanzia di approssimazione*, e quindi rivelatore di un errore, non di un
  risultato. La causa era che i numeri casuali venivano estratti *durante* la
  propagazione, così candidati diversi finivano valutati su realizzazioni
  aleatorie differenti e l'`argmax` sceglieva il più fortunato anziché il
  migliore. Siamo passati alla formulazione **live-edge con *frozen worlds***:
  i mondi possibili sono pre-campionati una volta per tutte e ogni candidato è
  valutato sulle stesse realizzazioni.

### 5. I controlli finali

Un risultato negativo può nascondere un difetto dei dati invece di una proprietà
del fenomeno, quindi abbiamo verificato due assunzioni implicite.

- **«La rete è troppo debole per contenere segnale.»** Confronto delle misure
  strutturali (clustering, cammino medio, modularità, esponente della
  distribuzione dei gradi stimato con il metodo di Clauset et al.) contro due
  **modelli nulli**: Erdős–Rényi e, soprattutto, il **configuration model**, che
  tiene fissa la sequenza esatta dei gradi — è il confronto rigoroso, perché in
  una rete con pochi hub molto connessi un clustering elevato potrebbe derivare
  dalla sola combinatoria dei gradi.
- **«Conta il tema, non la posizione.»** Le comunità di Louvain sono insiemi di
  identificativi privi di interpretazione: abbiamo attribuito a ciascuna
  un'etichetta tematica (campionando commenti dalla sola finestra del grafo e
  facendoli etichettare da un LLM con output vincolato a JSON) per poter chiedere
  se comunità tematicamente caratterizzate esibiscano tassi di persuasione
  diversi. L'analisi è descrittiva e non alimenta il classificatore.

---

## Struttura del codice

```
src/
├── common.py                percorsi, filtri, IO — condiviso da tutto
├── pipeline/                la pipeline principale, in ordine di step
│   ├── step1_units.py         path unit, rimozione OP, pulizia, finestre
│   ├── step2_graph.py         rete di interazione + blocco B2
│   ├── step2b_node2vec.py     embedding B2′
│   ├── step3_features.py      blocco B1
│   ├── step3b_semantic.py     blocco B3 (lessicale + Sentence-BERT)
│   ├── step5_train.py         ablazione appaiata + test di permutazione
│   ├── step5b_ablation_b3.py  ablazione separata dei sottoblocchi di B3
│   ├── step5c_ablation_nomic.py  replica con un secondo encoder semantico
│   └── export_colab.py        export per l'esecuzione su Colab
├── network/
│   ├── characterize.py        modelli nulli, small-world, legge di potenza
│   └── label_communities.py   etichettatura tematica delle comunità
├── influence/
│   ├── delta_graph.py         costruzione e diagnosi del grafo dei delta
│   └── maximize.py            IC/LT, frozen worlds, Greedy, CELF, RIS
└── stance/                    modulo LLM, in sospeso
```

Il modulo `src/stance/` (classificazione della postura argomentativa via LLM) è
rimasto **in sospeso** e non contribuisce ai risultati: il tier gratuito dell'API
si esauriva dopo circa un sesto delle unità, e il tempo necessario a completarlo
valeva meno di quello investito su node2vec e influence maximization. Il
[README del modulo](src/stance/README.md) spiega da dove ripartire.

I corpora grezzi (Winning Arguments Corpus, formato ConvoKit) non sono inclusi
nel repository: si scaricano dalla fonte originale.
