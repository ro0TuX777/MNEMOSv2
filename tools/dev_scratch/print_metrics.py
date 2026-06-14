import json

d = json.load(open('benchmarks/mg_test_2b_metrics.json'))
for density, res in d.items():
    print(f'=== {density} ===')
    unscored = res['unscored']
    scored = res['scored']
    print(f'Unscored - UsefulRate: {unscored["useful_candidate_rate"]:.2f}, HubSat: {unscored["hub_saturation"]:.2f}, p95: {unscored["p95_latency"]:.2f}ms')
    print(f'Scored   - UsefulRate: {scored["useful_candidate_rate"]:.2f}, HubSat: {scored["hub_saturation"]:.2f}, p95: {scored["p95_latency"]:.2f}ms')
    print(f'Safety   - GovLeak: {scored["gov_leakage"]}, LinLeak: {scored["lin_leakage"]}')
    print(f'MaxHubPct: {scored["max_single_hub_pct"]:.2f}')
    print(f'MissingSupportFound: {scored["missing_support_found"]} (total queries {scored["queries_run"]})')
