# MG-Test-3: EchoFrame Shadow Packet Injection

- `[x]` Research phase
  - `[x]` Review `run_mg_test_2c.py` to understand baseline behavior and GraphTier interactions
  - `[x]` Understand Candidate Envelope structure and how EchoFrame generates `final_results`
  - `[x]` Understand how to mock/simulate the generation of two packets: baseline and shadow
- `[x]` Implement `run_mg_test_3.py`
  - `[x]` Add `hub_penalty_floor = 0.2`, `score_threshold = 0.2`
  - `[x]` Generate Baseline EchoFrame packet
  - `[x]` Generate Shadow EchoFrame packet with appended graph candidates
  - `[x]` Append S# tags for graph candidates
  - `[x]` Collect required metrics
- `[x]` Run `run_mg_test_3.py`
  - `[x]` Generate `mg_test_3_metrics.json`
- `[x]` Verification
  - `[x]` Verify pass gates and stop gates
- `[x]` Artifacts Generation
  - `[x]` Write `mg_test_3_evaluation_report.md`
  - `[x]` Update `walkthrough.md`
