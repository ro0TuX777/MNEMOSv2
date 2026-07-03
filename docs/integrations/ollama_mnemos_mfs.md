# Ollama: MNEMOS MFS Adapter Setup

Status: `MNEMOS_OLLAMA_MFS_LOCAL_ADAPTER`

This guide makes MNEMOS available to local Ollama workflows through the MFS
boundary pattern. It is for hosts that use Ollama as the model runtime but do
not natively mount MCP tools.

If your Ollama-facing application already supports MCP, prefer the existing
MNEMOS MCP bridge in `mcp_servers/mnemos`. If it only calls Ollama's REST API,
use the adapter below.

## Boundary

The Ollama adapter is an R0-style context path:

- retrieves bounded evidence through `mnemos_sdk`;
- sends that evidence to Ollama `/api/chat`;
- returns the Ollama answer plus MNEMOS citations;
- does not write memory;
- does not alter MNEMOS retrieval behavior;
- does not enable or repair R1/R2 enforcement policy.

Claim boundary:

```text
MFS_OLLAMA_ADAPTER_R0_CONTEXT_ONLY
```

## 1. Start MNEMOS

Windows:

```powershell
cd G:\MNEMOS
docker compose up -d --build
curl http://localhost:8700/health
```

macOS:

```bash
python -m installer
docker compose -f docker-compose.generated.yml up -d --build
curl http://localhost:8700/health
```

On macOS, CPU mode is expected. The generated compose file should not contain
`runtime: nvidia`, and `.env.mnemos` should contain `MNEMOS_GPU_DEVICE=cpu`.

## 2. Start Ollama

Install and start Ollama, then pull a local chat model:

```bash
ollama pull llama3.1
ollama serve
```

If Ollama is already running as a background service, `ollama serve` may report
that the port is already in use. That is fine; the adapter expects
`http://localhost:11434` by default.

If your machine sets `OLLAMA_HOST` to a custom bind address, mirror that in
`OLLAMA_BASE_URL`. For example, `OLLAMA_HOST=0.0.0.0:7777` should use
`OLLAMA_BASE_URL=http://127.0.0.1:7777` for local adapter calls.

## 3. Ask From MNEMOS Evidence

Windows:

```powershell
$env:MNEMOS_BASE_URL = "http://localhost:8700"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
python tools/mnemos_ollama_chat.py `
  --model llama3.1 `
  --query "What is the current R1 evidence decision?"
```

macOS:

```bash
export MNEMOS_BASE_URL=http://localhost:8700
export OLLAMA_BASE_URL=http://localhost:11434
python tools/mnemos_ollama_chat.py \
  --model llama3.1 \
  --query "What is the current R1 evidence decision?"
```

For machine-readable output:

```bash
python tools/mnemos_ollama_chat.py --model llama3.1 --query "..." --json
```

The JSON output includes:

- `answer`
- `citations`
- `model`
- `claim_boundary`
- raw `ollama_response`

## 4. Optional Seed Data

If MNEMOS has an empty or noisy active collection, seed focused repo context
first:

```bash
python tools/seed_mnemos_repo_summaries.py
python tools/seed_mnemos_repo_context.py
```

## 5. Research Artifact Intake

For regular research workflows, use MNEMOS as an artifact-backed research
memory before asking Ollama broad synthesis questions. The intake tool supports
Markdown, text, code/data files, PDFs, and `.docx` files.

### Local UI

For a browser form instead of a long terminal command:

```powershell
python tools/mnemos_research_ui.py
```

Open:

```text
http://127.0.0.1:8788
```

The page lets you:

- set `MNEMOS_BASE_URL` and `OLLAMA_BASE_URL`;
- test both connections before intake;
- refresh locally available Ollama models into a dropdown;
- type a model name manually if it is not listed;
- upload PDFs, `.docx`, Markdown, text, code, or data files through the browser;
- set project, capability, status, tags, and output packet path;
- run the same `mnemos_research_intake.py` logic used by the CLI.

The UI does not provide chat. After intake, use the Open WebUI-compatible proxy
or `tools/mnemos_ollama_chat.py` for prompt sessions over the indexed evidence.

The UI is local-only by default (`127.0.0.1`) and stores uploaded files in the
OS temp directory unless `--upload-dir` is supplied:

```powershell
python tools/mnemos_research_ui.py --upload-dir C:\Users\vin\mnemos_research_uploads
```

macOS/Linux:

```bash
python tools/mnemos_research_ui.py --upload-dir ~/mnemos_research_uploads
```

### CLI

Example:

```powershell
python tools/mnemos_research_intake.py `
  "C:\Users\vin\Downloads\AI-Assisted Software Development Workflow.pdf" `
  docs\integrations\ollama_mnemos_mfs.md `
  --project MNEMOS `
  --capability "local research memory" `
  --status reviewed `
  --tag workflow `
  --tag ollama `
  --summarize-with-ollama `
  --ollama-model "hf.co/danchev/ibm-granite-docling-258M-GGUF:BF16" `
  --output docs\research\mnemos_ollama_research_memory_packet.md
```

On macOS:

```bash
python tools/mnemos_research_intake.py \
  ~/Downloads/AI-Assisted\ Software\ Development\ Workflow.pdf \
  docs/integrations/ollama_mnemos_mfs.md \
  --project MNEMOS \
  --capability "local research memory" \
  --status reviewed \
  --tag workflow \
  --tag ollama \
  --summarize-with-ollama \
  --ollama-model llama3.1 \
  --output docs/research/mnemos_ollama_research_memory_packet.md
```

The intake tool:

- extracts readable text from each artifact;
- chunks the text with source metadata;
- indexes chunks through `mnemos_sdk`;
- optionally asks Ollama to create a Markdown research intake packet;
- records `MFS_RESEARCH_INTAKE_R0_ARTIFACT_MEMORY` as the claim boundary.

Recommended metadata:

| Field | Example |
|---|---|
| `--project` | `MNEMOS`, `SAM`, `AIPAM` |
| `--capability` | `local research memory`, `agent planning`, `retrieval governance` |
| `--status` | `new`, `reviewed`, `promising`, `rejected`, `integrated` |
| `--tag` | `arxiv`, `github`, `paper`, `code`, `workflow` |

After intake, ask through the Ollama adapter:

```powershell
python tools/mnemos_ollama_chat.py `
  --model "hf.co/danchev/ibm-granite-docling-258M-GGUF:BF16" `
  --query "Which indexed artifacts suggest useful approaches for local research memory in MNEMOS?"
```

The goal is not to let Ollama decide what to build. The goal is to preserve
artifacts, summaries, risks, open questions, and source citations so future
implementation decisions have a durable evidence trail.

## 6. Open WebUI-Compatible Proxy

For regular chat in Open WebUI, start the MNEMOS proxy after MNEMOS and Ollama
are running:

```powershell
$env:MNEMOS_BASE_URL="http://127.0.0.1:8700"
$env:OLLAMA_BASE_URL="http://127.0.0.1:7777"

python tools/mnemos_ollama_openwebui_proxy.py --port 8790
```

macOS/Linux:

```bash
export MNEMOS_BASE_URL="http://127.0.0.1:8700"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"

python tools/mnemos_ollama_openwebui_proxy.py --port 8790
```

Then configure Open WebUI to use the proxy as an OpenAI-compatible provider:

```text
Base URL: http://127.0.0.1:8790/v1
API key: any local placeholder value, if the UI requires one
Model: choose one of the models returned by /v1/models
Prefix ID: mnemos, recommended when a direct Ollama connection is also enabled
```

When Open WebUI sends a prefixed model ID such as
`mnemos.hf.co/WSDW/Qwen2.5-7B-Instruct-Q4_K_M-GGUF:Q4_K_M`, the proxy strips
the `mnemos.` prefix before calling Ollama, while keeping the prefixed name in
the response for Open WebUI bookkeeping. Additional prefixes can be configured
with `MNEMOS_PROXY_MODEL_PREFIXES` as a comma-separated list.

The proxy also exposes Ollama-shaped endpoints for clients that expect Ollama's
native API:

```text
GET  http://127.0.0.1:8790/api/tags
POST http://127.0.0.1:8790/api/chat
```

For Open WebUI configurations that append Ollama routes to the OpenAI base URL,
the proxy also accepts:

```text
GET  http://127.0.0.1:8790/v1/api/tags
```

Chat requests are handled as:

1. extract the latest user message;
2. retrieve bounded MNEMOS evidence with the existing SDK defaults;
3. call the selected Ollama model with MNEMOS evidence injected;
4. return the answer plus a `mnemos` metadata block containing citations and
   the `MFS_OLLAMA_ADAPTER_R0_CONTEXT_ONLY` claim boundary.

The proxy accepts both non-streaming and streaming requests. Streaming is
compatibility streaming: MNEMOS retrieval and the Ollama call complete first,
then the proxy emits the completed answer as OpenAI Server-Sent Events or
Ollama JSON lines. This keeps the evidence boundary intact while supporting
Open WebUI clients that send `stream: true` by default.

Direct smoke test:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8790/v1/chat/completions" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{
    "model": "qwen3-coder-next:latest",
    "messages": [
      {
        "role": "user",
        "content": "Using indexed MNEMOS evidence, summarize the AI-assisted software development workflow."
      }
    ],
    "temperature": 0,
    "max_tokens": 700
  }'
```

## 7. MCP-Capable Ollama Hosts

Some Ollama front ends or agent hosts can call MCP tools while using Ollama as
their model backend. In that case, use the MNEMOS MCP bridge directly:

```bash
python tools/setup_mnemos_mcp_env.py
mcp_servers/mnemos/.venv/Scripts/python.exe mcp_servers/mnemos/server.py
```

On macOS, the bridge Python is:

```bash
mcp_servers/mnemos/.venv/bin/python mcp_servers/mnemos/server.py
```

See `docs/integrations/claude_desktop_mnemos_mcp.md` for the MCP bridge tools,
smoke checks, and troubleshooting. The MCP bridge and the Ollama adapter expose
the same MNEMOS memory boundary through different host surfaces.

## Troubleshooting

- If the adapter prints `MNEMOS returned no evidence`, MNEMOS was reachable but
  the current collection did not return supporting results.
- If the adapter cannot connect to MNEMOS, check `MNEMOS_BASE_URL` and
  `/health`.
- If the adapter cannot connect to Ollama, check `OLLAMA_BASE_URL` and run
  `curl http://localhost:11434/api/tags`.
- If the first call is slow, warm both systems: run a small MNEMOS search and
  `ollama run <model>` once before the real query.
- If your host supports MCP, do not duplicate tool wiring through this adapter;
  mount the MNEMOS MCP bridge and let the host decide when to call memory tools.
