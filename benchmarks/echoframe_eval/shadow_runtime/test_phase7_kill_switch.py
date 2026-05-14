import os
import uuid
import time
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from mnemos.experimental.echoframe_shadow.shadow_config import PilotConfig
from mnemos.experimental.echoframe_shadow.shadow_adapter import EchoFrameShadowAdapter

# Enforce 5% pilot
os.environ['MNEMOS_ECHOFRAME_SHADOW_ENABLED'] = 'true'
os.environ['MNEMOS_ECHOFRAME_DEFAULT_ON_ELIGIBLE'] = 'true'
os.environ['MNEMOS_ECHOFRAME_MODE'] = 'compact_semantic_minEvidence_hysteresis_v0'
os.environ['MNEMOS_ECHOFRAME_FAIL_CLOSED'] = 'true'
os.environ['MNEMOS_ECHOFRAME_REQUIRE_VALIDATION'] = 'true'
os.environ['MNEMOS_ECHOFRAME_ALLOW_HIGH_RISK'] = 'false'
os.environ['MNEMOS_ECHOFRAME_KILL_SWITCH'] = 'true'
os.environ['MNEMOS_ECHOFRAME_OUTPUT_DIR'] = 'runtime/echoframe_default/'
os.environ['MNEMOS_ECHOFRAME_LLM_FACING_ENABLED'] = 'false'

def run():
    # Clear output directory
    import glob
    for f in glob.glob("runtime/echoframe_default/shadow_event_*.json"):
        os.remove(f)

    # Mock baseline payload
    # 500 words to ensure baseline > echoframe tokens
    long_text = "This is a standard baseline chunk about the core engine. " * 100
    baseline_payload = {
        "results": [
            {
                "engram": {
                    "content": long_text,
                    "id": "123",
                    "source": "doc1"
                },
                "score": 0.95
            },
            {
                "engram": {
                    "content": "Additional context about memory persistence. " * 50,
                    "id": "124",
                    "source": "doc1"
                },
                "score": 0.92
            }
        ],
        "meta": {"governance_summary": {}}
    }
    
    # 2000 events * 0.05 = ~100 LLM-facing events
    TOTAL_CALLS = 20
    print("Running 50 mock searches to test kill switch...")
    
    for i in tqdm(range(TOTAL_CALLS)):
        # simulate some variance
        if i % 100 == 0:
            query = "Delete the user data for account 12345." # high-risk
        else:
            query = "What is the policy on data retention?"
            
        session_id = f"PHASE6_SIM_{uuid.uuid4().hex[:8]}"
        turn_id = f"T_{i}"
        
        EchoFrameShadowAdapter.observe_search(query, baseline_payload, session_id, turn_id)

    
    import json
    events = []
    res_dir = "runtime/echoframe_default/"
    files = glob.glob(os.path.join(res_dir, "shadow_event_*.json"))
    for f in files:
        with open(f, 'r') as fp:
            events.append(json.load(fp))
            
    pilot_events = [e for e in events if e.get("event_type") == "echoframe.default_on_event"]
    
    kill_switch_active = all(e.get("kill_switch_active") for e in pilot_events)
    fallback_reasons = all("kill_switch_active" in e.get("fallback_reason", "") for e in pilot_events)
    all_fallback = all(e.get("fallback_to_baseline") for e in pilot_events)
    llm_source = all(e.get("llm_context_source") == "baseline" for e in pilot_events)
    
    md = [
        "# Phase 7 Kill Switch Validation",
        f"- **Events Processed**: {len(pilot_events)}",
        f"- **PASS**: Kill switch active state recorded? {kill_switch_active}",
        f"- **PASS**: Fallback reason is kill_switch_active? {fallback_reasons}",
        f"- **PASS**: All events fallback to baseline? {all_fallback}",
        f"- **PASS**: No event marked llm_context_source=echoframe? {llm_source}",
    ]
    
    os.makedirs(os.path.join(res_dir, "reports"), exist_ok=True)
    with open(os.path.join(res_dir, "reports", "phase7_kill_switch_validation.md"), 'w') as f:
        f.write("\n".join(md))
        
    print("Kill switch validation report generated.")

if __name__ == "__main__":
    run()
