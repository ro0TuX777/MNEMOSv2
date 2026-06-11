import os
import sys
import json
import time
import requests
import argparse
from pathlib import Path

def setup_directories():
    os.makedirs("eval_results", exist_ok=True)
    os.makedirs("docs/reports", exist_ok=True)

def preflight_check(url: str):
    """Ensure the service and stats are reachable."""
    print(f"[*] Preflight check: {url}/v1/mnemos/stats")
    try:
        resp = requests.get(f"{url}/v1/mnemos/stats", timeout=5)
        if resp.status_code != 200:
            print("MNEMOS_INSTANCE_UNAVAILABLE")
            sys.exit(1)
        return resp.json()
    except requests.exceptions.RequestException:
        print("MNEMOS_INSTANCE_UNAVAILABLE")
        sys.exit(1)

def run_trial(url: str, interactive: bool, expect_kill_switch_disabled: bool):
    setup_directories()
    
    # 1. Preflight
    stats_before = preflight_check(url)
    mom_before = stats_before.get("stats", {}).get("derived_lane", {})
    
    print("\n[*] Starting PIT-8 Live Trial Harness...")
    metrics = {
        "latency_ms": {},
        "kill_switch_live_mode": "unknown",
        "boundary_tests": {},
        "telemetry_deltas": {}
    }
    raw_responses = {
        "_meta": {"sensitive_local_evaluation_artifact": True, "notice": "Do not expose."},
        "tests": {}
    }
    
    session = requests.Session()
    
    # Helper to track time and raw output
    def call_api(method, endpoint, **kwargs) -> tuple[requests.Response, float]:
        start = time.perf_counter()
        resp = session.request(method, f"{url}{endpoint}", timeout=10, **kwargs)
        latency = (time.perf_counter() - start) * 1000
        return resp, latency

    # ---------------------------------------------------------
    # Hard Gates
    # ---------------------------------------------------------
    
    # Gate 1: /api/v1/query with evaluation_mode=true returns HTTP 400
    print("[1] Test: /api/v1/query rejects evaluation_mode=true")
    resp, lat = call_api("POST", "/api/v1/query", json={"query": "test", "evaluation_mode": True})
    raw_responses["tests"]["api_v1_query_eval_mode"] = {"status": resp.status_code, "body": resp.text}
    metrics["latency_ms"]["api_v1_query_eval_mode"] = lat
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
    metrics["boundary_tests"]["api_v1_query_rejects_eval_mode"] = "PASS"

    # Gate 2: /api/v1/query returns zero derived facts
    print("[2] Test: /api/v1/query returns zero derived facts")
    resp, lat = call_api("POST", "/api/v1/query", json={"query": "test"})
    raw_responses["tests"]["api_v1_query_default"] = {"status": resp.status_code, "body": resp.text}
    metrics["latency_ms"]["api_v1_query_default"] = lat
    data = resp.json()
    assert len(data.get("derived_results", [])) == 0, "Leaked derived facts in /api/v1/query"
    metrics["boundary_tests"]["api_v1_query_zero_derived"] = "PASS"

    # Gate 3: /v1/mnemos/search returns zero derived facts
    print("[3] Test: /v1/mnemos/search returns zero derived facts")
    resp, lat = call_api("POST", "/v1/mnemos/search", json={"query": "test"})
    raw_responses["tests"]["v1_mnemos_search_default"] = {"status": resp.status_code, "body": resp.text}
    metrics["latency_ms"]["v1_mnemos_search_default"] = lat
    data = resp.json()
    assert len(data.get("derived_results", [])) == 0, "Leaked derived facts in /v1/mnemos/search"
    metrics["boundary_tests"]["v1_mnemos_search_zero_derived"] = "PASS"

    # Gate 4: /api/v1/evaluate_derived_shadow requires both flags
    print("[4] Test: /api/v1/evaluate_derived_shadow missing flags")
    headers = {"X-Client-Id": "eval_dashboard"}
    resp, lat = call_api("POST", "/api/v1/evaluate_derived_shadow", headers=headers, json={"evaluation_mode": True})
    raw_responses["tests"]["evaluate_derived_shadow_missing_flags"] = {"status": resp.status_code, "body": resp.text}
    metrics["latency_ms"]["evaluate_derived_shadow_missing_flags"] = lat
    assert resp.status_code in (400, 503), f"Expected 400 (or 503 if killed), got {resp.status_code}"
    if resp.status_code == 400:
        metrics["boundary_tests"]["evaluate_derived_shadow_missing_flags"] = "PASS"

    # Gate 5: unauthorized X-Client-Id returns HTTP 403
    print("[5] Test: /api/v1/evaluate_derived_shadow unauthorized client")
    headers = {"X-Client-Id": "hacker"}
    resp, lat = call_api("POST", "/api/v1/evaluate_derived_shadow", headers=headers, json={"evaluation_mode": True, "include_derived_facts": True})
    raw_responses["tests"]["evaluate_derived_shadow_unauthorized"] = {"status": resp.status_code, "body": resp.text}
    metrics["latency_ms"]["evaluate_derived_shadow_unauthorized"] = lat
    assert resp.status_code in (403, 503), f"Expected 403 (or 503 if killed), got {resp.status_code}"
    if resp.status_code == 403:
        metrics["boundary_tests"]["evaluate_derived_shadow_unauthorized"] = "PASS"

    # Gate 6: Success path and kill-switch
    print("[6] Test: /api/v1/evaluate_derived_shadow success path")
    headers = {"X-Client-Id": "eval_dashboard"}
    resp, lat = call_api("POST", "/api/v1/evaluate_derived_shadow", headers=headers, json={"evaluation_mode": True, "include_derived_facts": True, "query": "hello"})
    raw_responses["tests"]["evaluate_derived_shadow_success"] = {"status": resp.status_code, "body": resp.text}
    metrics["latency_ms"]["evaluate_derived_shadow_success"] = lat

    if expect_kill_switch_disabled:
        assert resp.status_code == 503, "Expected 503 when kill switch is disabled"
        metrics["kill_switch_live_mode"] = "manual_restart_verified"
        metrics["boundary_tests"]["evaluate_derived_shadow_success_path"] = "SKIPPED_DUE_TO_KILL_SWITCH"
        shadow_payload = None
    else:
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        metrics["kill_switch_live_mode"] = "active_enabled_unit_test_evidence_only"
        
        data = resp.json()
        assert "shadow_evaluation" in data, "Missing shadow_evaluation block"
        rendered_block = data["shadow_evaluation"]["rendered_block"]
        if "[MNEMOS-DERIVED]" not in rendered_block:
            print(f"DEBUG rendered_block:\n{rendered_block}")
            assert False, "Authority label missing in shadow output"
        metrics["boundary_tests"]["evaluate_derived_shadow_success_path"] = "PASS"
        shadow_payload = data["shadow_evaluation"]

    # ---------------------------------------------------------
    # Telemetry Deltas
    # ---------------------------------------------------------
    stats_after = preflight_check(url)
    mom_after = stats_after.get("stats", {}).get("derived_lane", {})
    
    for key in mom_after.keys():
        before_val = mom_before.get(key, 0)
        after_val = mom_after.get(key, 0)
        metrics["telemetry_deltas"][key] = after_val - before_val
        
    assert metrics["telemetry_deltas"].get("query.default_retrieval.derived_fact_count", 0) == 0, "Leaked derived facts in default retrieval"

    # ---------------------------------------------------------
    # Save Outputs
    # ---------------------------------------------------------
    with open("eval_results/pit_8_live_trial_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    with open("eval_results/pit_8_live_trial_raw_responses.json", "w") as f:
        json.dump(raw_responses, f, indent=2)

    with open("eval_results/pit_8_live_trial_report.md", "w") as f:
        f.write("# PIT-8 Live Trial Report\n\n")
        f.write("**Status**: EXPERIMENT COMPLETED\n")
        f.write("**Kill Switch State**: {}\n\n".format(metrics["kill_switch_live_mode"]))
        f.write("> **NOTICE**: This harness performed read-only queries and did NOT mutate the server state. No automatic promotion, no Candidate Envelope mixing, and no schema extraction was executed.\n\n")
        
        f.write("## Boundary Tests\n")
        for k, v in metrics["boundary_tests"].items():
            f.write(f"- `{k}`: **{v}**\n")
            
        f.write("\n## Telemetry Deltas\n")
        f.write("```json\n")
        f.write(json.dumps(metrics["telemetry_deltas"], indent=2))
        f.write("\n```\n")

        f.write("\n## Latency\n")
        for k, v in metrics["latency_ms"].items():
            f.write(f"- `{k}`: {v:.2f} ms\n")
            
        if shadow_payload:
            f.write("\n## Qualitative Review (Shadow Packet)\n")
            f.write("```markdown\n")
            f.write(shadow_payload["rendered_block"])
            f.write("\n```\n")

    print("\n[+] PIT-8 Live Trial Complete. Results written to eval_results/")

    # Interactive Review
    if interactive and shadow_payload:
        print("\n=== INTERACTIVE REVIEW ===")
        print(shadow_payload["rendered_block"])
        input("Press Enter to complete review...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8700", help="MNEMOS server URL")
    parser.add_argument("--interactive-review", action="store_true", help="Prompt operator for review")
    parser.add_argument("--expect-kill-switch-disabled", action="store_true", help="Assert that the server is returning 503")
    args = parser.parse_args()
    
    run_trial(args.url, args.interactive_review, args.expect_kill_switch_disabled)
