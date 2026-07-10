Here's the full picture. Six services, two storage layers, one host process (Ollama), and two main flows: putting knowledge in (intake) and getting answers out with proof (chat + receipts).

Topology

                            ┌──────────────────────┐
                            │     YOUR BROWSER     │
                            └──────┬────────┬──────┘
                     :8788         │        │         :8088
               ┌───────────────────┘        └───────────────────┐
               ▼                                                ▼
 ┌───────────────────────────┐                   ┌───────────────────────────┐
 │  research-ui       :8788  │                   │  open-webui        :8088  │
 │  (mnemos compose stack)   │                   │  (SEPARATE container)     │
 │                           │                   │                           │
 │  • upload PDFs/docs       │                   │  • chat interface only    │
 │  • extract → chunk →      │                   │  • speaks OpenAI API      │
 │    index into MNEMOS      │                   │  • knows nothing about    │
 │  • /evidence receipt      │                   │    MNEMOS directly        │
 │    browser                │                   └─────────────┬─────────────┘
 └──────┬─────────────▲──────┘                                 │
        │             │                            host.docker.internal:8790
        │ index docs  │ reads receipts                         ▼
        │ (mnemos     │ (./logs mount)            ┌───────────────────────────┐
        │  :8700)     │                           │  openwebui-proxy   :8790  │
        │             └───────────────────────────│  the "receipt factory"   │
        │                                         │                           │
        │            ┌────────────────────────────│  1. retrieve evidence     │
        │            │   retrieve evidence        │     from MNEMOS           │
        ▼            ▼   (mnemos :8700)           │  2. send evidence+question│
 ┌───────────────────────────┐                    │     to Ollama             │
 │  mnemos-service    :8700  │                    │  3. append evidence footer│
 │  the memory engine        │                    │  4. write receipt JSON    │
 │                           │                    └───────┬───────────┬───────┘
 │  • embeds text (GPU)      │                            │           │
 │  • stores + retrieves     │        host.docker.        │           ▼
 │  • audit logging          │        internal:7777       │   ./logs/evidence_receipts
 └──────┬────────────┬───────┘                            ▼   (host folder, shared
        │            │                          ┌─────────────────┐  with research-ui)
        ▼            ▼                          │  OLLAMA (host   │
 ┌────────────┐ ┌────────────┐                  │  process, NOT   │
 │  qdrant    │ │  postgres  │                  │  a container)   │
 │  :6333     │ │  :5432     │                  │  :7777          │
 │  vector    │ │  audit /   │                  │  runs the LLM   │
 │  search    │ │  metadata  │                  └─────────────────┘
 └────────────┘ └────────────┘
What each container is for
Container	Role in one sentence
qdrant	Vector database — stores the embeddings of your document chunks so "find text similar to my question" is fast.
postgres	Relational database — MNEMOS's audit trail and structured metadata; the paper ledger next to qdrant's search index.
mnemos	The memory engine — the only service that touches the databases; turns text into embeddings (GPU), indexes chunks, and answers retrieval queries over HTTP :8700.
research-ui	Your intake desk — upload PDFs, it extracts/chunks/labels them and pushes them into mnemos; also hosts the /evidence receipt browser.
openwebui-proxy	The translator with a notary stamp — pretends to be an OpenAI-compatible model server so Open WebUI can use it, but actually fetches evidence from mnemos, hands it to Ollama, and stamps every answer with an evidence receipt.
open-webui	Just the chat window — a generic UI that thinks it's talking to a normal model API; all the MNEMOS magic happens in the proxy behind it.
The two flows
Flow 1 — Intake (knowledge goes in):


You upload paper.pdf at :8788
  → research-ui extracts text (pypdf, OCR fallback), chunks it,
    attaches metadata (source, pages, hashes)
  → POST to mnemos :8700
  → mnemos embeds each chunk on the GPU
  → vectors land in qdrant, audit entries in postgres
Flow 2 — Chat with receipts (knowledge comes out):


You ask a question in open-webui at :8088
  → open-webui sends an OpenAI-style request to the proxy :8790
  → proxy asks mnemos :8700 "what chunks are relevant?"
       → mnemos searches qdrant, returns evidence + sources
  → proxy sends evidence + your question to Ollama :7777 (host)
  → Ollama generates the answer FROM that evidence
  → proxy appends the evidence footer and writes a receipt JSON
    to ./logs/evidence_receipts
  → you read the receipt later in research-ui /evidence
Two details worth remembering
Two networks, one bridge. The five mnemos-stack containers talk to each other by service name (mnemos:8700, qdrant:6333). Open-webui lives in a separate Docker network, so it reaches the proxy the only way it can — back out through your host via host.docker.internal:8790. Same trick for Ollama, which isn't a container at all: everything reaches it at host.docker.internal:7777.

The receipts live on your host disk, not inside any container. ./logs/evidence_receipts in G:\MNEMOS is mounted into both the proxy (writer) and research-ui (reader). That's why your 46 receipts survived every container restart — and why the two containers see the same files.

The key architectural idea: open-webui never knows MNEMOS exists, and mnemos never knows chat exists. The proxy is the only piece that knows both worlds, which keeps the memory engine's boundary clean — exactly the MFS_OLLAMA_ADAPTER_R0_CONTEXT_ONLY claim boundary stamped in each receipt.
