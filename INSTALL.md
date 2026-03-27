# MNEMOS — Installation Guide

## Prerequisites

- **Python** 3.10+
- **Docker** (optional, for containerised deployment)
- **pip** (Python package manager)

---

## Option 1: Local Development

### 1. Clone / Copy the Project

```bash
cd /path/to/MNEMOS
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**

| Package | Purpose |
|---|---|
| `flask` | REST API framework |
| `chromadb` | Tier 1 vector store |
| `lancedb` | Tier 2 hybrid store |
| `sentence-transformers` | Embedding models (MiniLM, ColBERT) |
| `numpy`, `scipy` | TurboQuant compression |
| `gunicorn` | Production WSGI server |

### 4. Configure

```bash
# Copy the example config
cp .env.example .env

# Edit as needed
notepad .env   # Windows
nano .env      # Linux
```

Key settings:

| Variable | Default | Description |
|---|---|---|
| `MNEMOS_TIERS` | `chromadb` | Active tiers: `chromadb`, `lancedb`, `colbert` (comma-separated) |
| `MNEMOS_QUANT_BITS` | `4` | TurboQuant bit-width (0 = disabled, 1–4) |
| `MNEMOS_AUDIT_ENABLED` | `true` | Enable forensic audit ledger |
| `MNEMOS_PORT` | `8700` | API port |
| `MNEMOS_TOKEN` | *(empty)* | Optional bearer auth token |

### 5. Run

```bash
# Development
python -m service.app

# Production
gunicorn -w 2 -b 0.0.0.0:8700 service.app:app
```

### 6. Verify

```bash
curl http://localhost:8700/health
# → {"status": "ok", "service": "mnemos-service", "contract_version": "v1"}

curl http://localhost:8700/v1/mnemos/capabilities
# → full capabilities payload

# Or use the health audit tool
python tools/mnemos_health_audit.py
```

---

## Option 2: Docker

### 1. Build and Run

```bash
cd /path/to/MNEMOS
docker compose up -d --build
```

### 2. Verify

```bash
docker logs mnemos-service
curl http://localhost:8700/health

# Run health audit
python tools/mnemos_health_audit.py
```

### 3. Customise via Environment

Edit `docker-compose.yml` or pass env vars:

```bash
docker compose up -d -e MNEMOS_TIERS=chromadb,colbert -e MNEMOS_QUANT_BITS=4
```

### 4. Persistent Data

Data is stored in `./data/` (mounted as a Docker volume). This includes:
- `data/chroma/` — ChromaDB index
- `data/lance/` — LanceDB tables
- `data/colbert/` — ColBERT multi-vector index
- `data/audit.db` — Forensic audit ledger

---

## Option 3: Integrate into an Existing Application

### 1. Onboard (generates boundary adapter, env template, quickstart)

```bash
python tools/mnemos_onboard.py --target /path/to/your-app
```

### 2. Add MNEMOS to your project's `docker-compose.yml`

```yaml
mnemos:
  build: ./MNEMOS                       # or image: mnemos-service:latest
  ports:
    - "8700:8700"
  volumes:
    - ./data/mnemos:/app/data
  environment:
    - MNEMOS_TIERS=chromadb
    - MNEMOS_QUANT_BITS=4
    - MNEMOS_AUDIT_ENABLED=true
  restart: unless-stopped
```

### 3. Use the Boundary SDK

```python
from mnemos_sdk import MnemosClient, MnemosConfig

client = MnemosClient(MnemosConfig.from_env())
client.wait_until_ready()

# Index a document
client.index([{"content": "Hello world", "source": "app"}])

# Search
hits = client.search("hello", top_k=5)
for hit in hits:
    print(f"  [{hit.score:.3f}] {hit.engram['content'][:80]}")
```

---

## Running Tests

```bash
pip install pytest
cd /path/to/MNEMOS
python -m pytest tests/ -v
```

Expected: **31 passed**.

---

## Operational Tools

| Tool | Usage |
|---|---|
| Health audit | `python tools/mnemos_health_audit.py` |
| Contract diff | `python tools/contract_diff.py --old service/contract.json --new contracts/v2.json --mode both` |
| CI gates | `python tools/mnemos_ci_gates.py --run-health-audit --run-container-build` |
| Cutover scaffold | `python tools/mnemos_cutover_scaffold.py --app my-app` |
| Onboard consumer | `python tools/mnemos_onboard.py --target /path/to/app` |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: chromadb` | Run `pip install -r requirements.txt` |
| `FTS5 not supported` warning | Upgrade SQLite or ignore (basic search still works) |
| ChromaDB slow on first run | Model download — `all-MiniLM-L6-v2` (~80 MB, one-time) |
| Port 8700 in use | Change `MNEMOS_PORT` in `.env` or `docker-compose.yml` |
| Permission denied on `data/` | Ensure the data directory is writable: `mkdir -p data && chmod 755 data` |
