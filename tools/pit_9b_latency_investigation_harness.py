import os
import sys
import json
import time
import requests
import argparse
import statistics

def setup_directories():
    os.makedirs("eval_results", exist_ok=True)
    os.makedirs("docs/reports", exist_ok=True)

def preflight_check(url: str):
    try:
        resp = requests.get(f"{url}/v1/mnemos/stats", timeout=5)
        if resp.status_code != 200:
            sys.exit(1)
        return resp.json()
    except requests.exceptions.RequestException:
        sys.exit(1)

def run_trial(url: str):
    setup_directories()
    stats_before = preflight_check(url)
    mom_before = stats_before.get("stats", {}).get("derived_lane", {})

    print("\n[*] Starting PIT-9B Latency Investigation Harness...")
    session = requests.Session()
    headers = {"X-Client-Id": "eval_dashboard"}
    
    metrics = {
        "warmup_request_latency_ms": 0.0,
        "cold_start_excluded": True,
        "burst_total_p50": 0.0,
        "burst_total_p95": 0.0,
        "burst_total_p99": 0.0,
        "suspected_latency_source": "",
        "stage_averages_ms": {},
        "boundary_tests": {},
        "telemetry_deltas": {}
    }
    
    # 0. Warm-Up Request to isolate CrossEncoder cold-start
    print("\n[+] Executing Warm-Up Request (Cold-Start Isolation)")
    start = time.perf_counter()
    resp = session.post(f"{url}/v1/mnemos/search", json={"query": "warm-up"})
    metrics["warmup_request_latency_ms"] = (time.perf_counter() - start) * 1000
    print(f"    Warm-up completed in {metrics['warmup_request_latency_ms']:.2f}ms")

    # 1. Automated Burst Latency (Load test bounded)
    print("\n[+] Phase 1: Automated Burst for Latency (p50/p90/p95/p99)")
    stage_samples = {
        "auth_whitelist_check_ms": [],
        "default_search_ms": [],
        "search_derived_ms": [],
        "governance_ledger_check_ms": [],
        "shadow_serializer_ms": [],
        "evaluation_renderer_ms": [],
        "telemetry_stats_update_ms": [],
        "response_serialization_ms": [],
        "total_request_ms": []
    }
    
    latencies = []
    
    for i in range(15):
        start = time.perf_counter()
        resp = session.post(f"{url}/api/v1/evaluate_derived_shadow", headers=headers, json={"evaluation_mode": True, "include_derived_facts": True, "query": "burst test"})
        if resp.status_code == 200:
            data = resp.json()
            stages = data.get("shadow_evaluation", {}).get("stage_latencies_ms", {})
            if not stages:
                print(f"DEBUG NO STAGES: {data}")
            for k, v in stages.items():
                if k in stage_samples:
                    stage_samples[k].append(v)
            if "total_request_ms" in stages:
                latencies.append(stages["total_request_ms"])
    
    if latencies:
        latencies.sort()
        metrics["burst_total_p50"] = latencies[len(latencies)//2]
        metrics["burst_total_p90"] = latencies[int(len(latencies)*0.90)]
        metrics["burst_total_p95"] = latencies[int(len(latencies)*0.95)]
        metrics["burst_total_p99"] = latencies[int(len(latencies)*0.99)]
        print(f"    p50: {metrics['burst_total_p50']:.2f}ms, p95: {metrics['burst_total_p95']:.2f}ms")

    for k, v in stage_samples.items():
        if v:
            metrics["stage_averages_ms"][k] = sum(v) / len(v)
            
    # Identify suspected source if warm-up was slow
    if metrics["warmup_request_latency_ms"] > 1000 and metrics["burst_total_p95"] < 500:
        metrics["suspected_latency_source"] = "default_search_ms (cold-start initialization)"
    elif metrics["stage_averages_ms"].get("default_search_ms", 0) > 500:
        metrics["suspected_latency_source"] = "default_search_ms"
    elif metrics["stage_averages_ms"].get("search_derived_ms", 0) > 500:
        metrics["suspected_latency_source"] = "search_derived_ms"
    else:
        metrics["suspected_latency_source"] = "none (latency resolved)"

    # 2. Boundary Integrity Check
    print("\n[+] Phase 2: Boundary Integrity Checks")
    resp = session.post(f"{url}/api/v1/query", json={"query": "test"})
    metrics["boundary_tests"]["default_retrieval_leakage_count_zero"] = len(resp.json().get("derived_results", [])) == 0

    resp = session.post(f"{url}/api/v1/query", json={"query": "test", "evaluation_mode": True})
    metrics["boundary_tests"]["production_prompt_leakage_count_zero"] = resp.status_code == 400

    # 3. Telemetry Verification
    stats_after = preflight_check(url)
    mom_after = stats_after.get("stats", {}).get("derived_lane", {})
    for key in mom_after.keys():
        metrics["telemetry_deltas"][key] = mom_after.get(key, 0) - mom_before.get(key, 0)
    
    assert metrics["telemetry_deltas"].get("query.default_retrieval.derived_fact_count", 0) == 0, "Default retrieval leaked derived facts!"

    # 5. Output Generation
    with open("eval_results/pit_9b_latency_profile.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("eval_results/pit_9b_latency_report.md", "w") as f:
        f.write("# PIT-9B Latency Investigation Report\n\n")
        f.write("## 1. Diagnostics\n")
        f.write(f"- Warm-up Request Latency: {metrics['warmup_request_latency_ms']:.2f} ms\n")
        f.write(f"- Suspected Source: **{metrics['suspected_latency_source']}**\n")
        f.write(f"- Cold Start Excluded from Burst: {metrics['cold_start_excluded']}\n")
        
        f.write("\n## 2. Server-Side Stage Latencies (Average)\n")
        for k, v in metrics["stage_averages_ms"].items():
            f.write(f"- `{k}`: {v:.2f} ms\n")
            
        f.write("\n## 3. Burst Latency Percentiles\n")
        f.write(f"- p50: {metrics.get('burst_total_p50', 0):.2f} ms\n")
        f.write(f"- p90: {metrics.get('burst_total_p90', 0):.2f} ms\n")
        f.write(f"- p95: {metrics.get('burst_total_p95', 0):.2f} ms\n")
        f.write(f"- p99: {metrics.get('burst_total_p99', 0):.2f} ms\n")

        f.write("\n## 4. Boundary Enforcement\n")
        for k, v in metrics["boundary_tests"].items():
            f.write(f"- `{k}`: **{'PASS' if v else 'FAIL'}**\n")

    print("\n[+] PIT-9B Harness Complete. Outputs written to eval_results/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8700")
    args = parser.parse_args()
    run_trial(args.url)
