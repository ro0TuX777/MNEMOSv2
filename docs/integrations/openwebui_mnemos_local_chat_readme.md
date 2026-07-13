# MNEMOS Local Chat With Ollama And Open WebUI

Status: `MNEMOS_OPENWEBUI_LOCAL_CHAT_R0`

This guide is the end-to-end user path for local research chat:

```text
Research files -> MNEMOS intake -> MNEMOS retrieval -> Ollama model -> Open WebUI chat
```

MNEMOS is the evidence source. Ollama is the model runtime. Open WebUI is the
chat interface.

## What Gets Automated

The helper script can:

- check MNEMOS at `http://127.0.0.1:8700`;
- detect Ollama from `OLLAMA_BASE_URL`, `OLLAMA_HOST`, or `http://127.0.0.1:11434`;
- start the MNEMOS Research Intake UI at `http://127.0.0.1:8788`;
- start the MNEMOS Open WebUI proxy at `http://127.0.0.1:8790`;
- print the Open WebUI settings to enter.

The helper script does not edit Open WebUI settings for you. Open WebUI keeps
those settings inside its own app database, so the user must enter them in the
Open WebUI admin page.

## What The User Still Does

The user must:

- start MNEMOS itself if it is not already running;
- start Ollama and make sure the desired model is pulled;
- start or install Open WebUI;
- enter the MNEMOS proxy connection in Open WebUI;
- choose the `mnemos...` model in chat;
- upload/select research artifacts in the MNEMOS intake page.

## Start MNEMOS And Ollama

MNEMOS must be reachable:

```powershell
Invoke-RestMethod http://127.0.0.1:8700/health
```

Ollama must be reachable. On this Windows setup, Ollama is commonly on port
`7777`:

```powershell
$env:OLLAMA_BASE_URL="http://127.0.0.1:7777"
Invoke-RestMethod "$env:OLLAMA_BASE_URL/api/tags"
```

Default Ollama installs often use:

```text
http://127.0.0.1:11434
```

## Start The MNEMOS Local Chat Helpers

From the MNEMOS repository:

```powershell
cd <path-to-MNEMOS>
$env:MNEMOS_BASE_URL="http://127.0.0.1:8700"
$env:OLLAMA_BASE_URL="http://127.0.0.1:7777"

python tools/start_mnemos_openwebui_stack.py --open-browser
```

For default Ollama on macOS/Linux:

```bash
export MNEMOS_BASE_URL="http://127.0.0.1:8700"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"

python tools/start_mnemos_openwebui_stack.py --open-browser
```

The script keeps the services it starts alive until you press `Ctrl+C`.

## Do We Still Need The Research Intake Page?

Yes.

Open WebUI is the chat surface, but it does not decide which PDFs, Markdown
files, Word documents, code files, or GitHub notes should become MNEMOS memory.
The Research Intake UI is the user-friendly way to put artifacts into MNEMOS
before asking questions.

Open:

```text
http://127.0.0.1:8788
```

Use it to:

- upload PDFs, `.docx`, Markdown, text, code, or data files;
- set metadata such as project, capability, status, and tags;
- optionally ask Ollama to create a research intake packet;
- index the extracted chunks into MNEMOS.

Recommended metadata for research workflows:

```text
Project: MNEMOS, SAM, or the target local project
Capability: AI-assisted software development workflow, local research memory, agent planning, etc.
Status: new, reviewed, promising, rejected, or integrated
Tags: workflow, pdf, arxiv, github, code, research
MNEMOS timeout seconds: 180 to 300 for large PDFs
Index batch size: 25, or 10/5 for large PDFs that time out
```

The Ollama model field on the intake page matters only when
`Summarize with Ollama` is checked. Indexing itself writes the artifact chunks
to MNEMOS.

## Configure Open WebUI

If Open WebUI is not installed yet, one Docker option is:

```powershell
docker run -d `
  --name open-webui `
  -p 8088:8080 `
  -v open-webui:/app/backend/data `
  --restart unless-stopped `
  ghcr.io/open-webui/open-webui:main
```

Open:

```text
http://127.0.0.1:8088
```

In Open WebUI:

```text
Admin Panel / Admin Settings
-> Settings
-> Connections
-> OpenAI
-> Add Connection
```

Use:

```text
Connection Type: Local
URL / Base URL: http://host.docker.internal:8790/v1
API Key: mnemos-local
Prefix ID: mnemos
Model IDs: leave empty
Headers: leave empty
```

Why `host.docker.internal`? Open WebUI runs inside Docker. From inside the
container, `127.0.0.1` means the container itself, not the Windows/macOS/Linux
host.

If you also keep a direct Ollama connection in Open WebUI, set it to the real
Ollama port:

```text
http://host.docker.internal:7777
```

or, for default Ollama:

```text
http://host.docker.internal:11434
```

Use the `mnemos...` prefixed model when you want MNEMOS-backed answers. Use the
direct Ollama model when you want normal non-MNEMOS chat.

## Validate End To End

In Open WebUI, choose a model with the `mnemos` prefix, for example:

```text
mnemos.hf.co/WSDW/Qwen2.5-7B-Instruct-Q4_K_M-GGUF:Q4_K_M
```

Ask:

```text
Using only MNEMOS evidence, what workflow did I index for AI-assisted software development? Cite the supporting sources.
```

A successful answer should:

- mention `MNEMOS_EVIDENCE` or clearly answer from supplied evidence;
- include bracket citations such as `[1]`, `[2]`;
- cite source paths from indexed artifacts;
- include a deterministic `MNEMOS Evidence Used` footer with source paths,
  scores, engram IDs, and the R0 boundary;
- include a local `MNEMOS Evidence Receipt` link for inspecting the evidence
  block that was supplied to Ollama;
- not invoke Open WebUI tools such as `query_knowledge_bases` or
  `search_knowledge_files`.

If the answer says no evidence was found, the chat path is probably working but
the right artifact has not been indexed or retrieved. Return to the intake page,
index the target files with clear metadata, and retry.

## Evidence Footers And Receipts

MNEMOS-backed answers append a proxy-generated footer. The footer is not written
by the model; it is added after MNEMOS retrieval and Ollama generation so Open
WebUI users can see the evidence layer that would otherwise be hidden in API
metadata.

Example:

```text
---
MNEMOS Evidence Used
[1] source=C:\...\paper.pdf
    score=0.8123
    engram_id=research::abc123

MNEMOS Evidence Receipt: http://127.0.0.1:8790/evidence/chatcmpl-mnemos-...

Boundary:
MFS_OLLAMA_ADAPTER_R0_CONTEXT_ONLY
```

The receipt page shows:

- the user query;
- the selected and actual Ollama model IDs;
- the citations;
- the evidence block sent to Ollama;
- retrieval metadata reported by MNEMOS;
- the claim boundary.

Receipts are local observability artifacts. They are stored under:

```text
logs/evidence_receipts/
```

Override the location if needed:

```powershell
$env:MNEMOS_EVIDENCE_RECEIPT_DIR="<path-to-receipts>"
```

To turn footers off:

```powershell
$env:MNEMOS_PROXY_FOOTER="off"
```

Open WebUI background title/task prompts are automatically suppressed so chat
titles are not polluted with evidence footers.

The Research Intake UI also includes an evidence browser:

```text
http://127.0.0.1:8788/evidence
```

Use it to review recent receipts, open a receipt detail page, and inspect the
evidence graph:

```text
User Query -> Retrieved Evidence Chunks -> Source Files / Metadata -> Model Answer With Citations
```

This is an evidence graph, not a reasoning graph. It shows what MNEMOS
retrieved and supplied to the local model.

## Troubleshooting

If Open WebUI says `Server Connection Error`, check:

```powershell
Invoke-RestMethod http://127.0.0.1:8790/health
Invoke-RestMethod http://127.0.0.1:8790/v1/api/tags
```

If the proxy works from the host but Open WebUI cannot reach it, confirm Open
WebUI uses:

```text
http://host.docker.internal:8790/v1
```

If the model appears with a prefix but returns a blank answer, update MNEMOS to
include the proxy prefix-normalization fixes and restart the proxy.

If Open WebUI invokes `query_knowledge_bases` or `search_knowledge_files`, it is
using Open WebUI's own knowledge tools rather than MNEMOS. Disable those tools
for the validation chat and select the `mnemos...` model.

If large PDFs time out during intake, use:

```text
MNEMOS timeout seconds: 300
Index batch size: 10
```

If still slow, reduce batch size to `5`.

## Boundary

This is an R0 context path:

```text
MFS_OLLAMA_ADAPTER_R0_CONTEXT_ONLY
```

It retrieves MNEMOS evidence and asks Ollama to answer from that evidence. It
does not alter MNEMOS retrieval policy, write chat answers back to memory, or
enable R1/R2 enforcement.
