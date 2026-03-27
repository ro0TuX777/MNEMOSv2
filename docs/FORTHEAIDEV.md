You are integrating or operating the MNEMOS memory service for an application.

Context:
- MNEMOS path: <MNEMOS_PATH>
- Consumer app path: <APP_PATH> (if integrating)
- Goal: provide the application with containerised, contract-governed memory and retrieval via MNEMOS.

Step 1: Read MNEMOS docs (required)
- <MNEMOS_PATH>/README.md
- <MNEMOS_PATH>/INSTALL.md
- <MNEMOS_PATH>/Use Cases.md
- <MNEMOS_PATH>/service/contract.json
- <MNEMOS_PATH>/mnemos_sdk/client.py (skim public API)

Step 2: Decide operation mode
- **Standalone**: MNEMOS runs independently, consumer calls via SDK or HTTP.
- **Composed**: MNEMOS is added to the consumer's docker-compose stack.

Step 3: Run onboarding (if integrating into an app)
- python <MNEMOS_PATH>/tools/mnemos_onboard.py --target <APP_PATH> --force
- Review generated artifacts in <APP_PATH>/mnemos_integration/.

Step 4: Start MNEMOS
- Docker: `docker compose up -d --build` (from <MNEMOS_PATH>)
- Local: `pip install -r requirements.txt && python -m service.app`

Step 5: Validate
- python <MNEMOS_PATH>/tools/mnemos_health_audit.py
- Confirm /health returns HTTP 200.
- Confirm /v1/mnemos/capabilities returns contract_version, status, and tier list.

Step 6: Wire the consumer app
For each integration point in the consumer app:
- Import the boundary: `from mnemos_sdk import MnemosClient, MnemosConfig`
- Configure via env vars: `MNEMOS_BASE_URL`, `MNEMOS_TOKEN`
- Use `client.wait_until_ready()` at app startup
- Use `client.index()` to store, `client.search()` to recall
- Handle degraded/unavailable status from the response envelope

Step 7: Produce integration summary (required)
Provide:
1. What you executed and artifacts generated
2. MNEMOS configuration (tiers, compression, port)
3. Integration points in consumer app (which modules call MNEMOS)
4. Boundary adapter wiring (env vars, readiness, error handling)
5. Verification results (health audit, smoke test)
6. Next 5 actions with exact commands
7. Risks, assumptions, open questions

Constraints:
- Always use the boundary SDK, never raw HTTP from consumer code.
- Always call wait_until_ready() before first use.
- Handle unavailable/degraded status explicitly — never silently ignore errors.
- Prefer env-var-driven configuration over hardcoded values.
- Run health audit after any configuration or deployment change.
