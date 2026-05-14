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
os.environ['MNEMOS_ECHOFRAME_LLM_FACING_ENABLED'] = 'true'
os.environ['MNEMOS_ECHOFRAME_LLM_FACING_MODE'] = 'compact_semantic_minEvidence_hysteresis_v0'
os.environ['MNEMOS_ECHOFRAME_LLM_FACING_SAMPLE_RATE'] = '0.05'
os.environ['MNEMOS_ECHOFRAME_LLM_FACING_OUTPUT_DIR'] = 'runtime/echoframe_pilot/'
os.environ['MNEMOS_ECHOFRAME_LLM_FACING_FAIL_CLOSED'] = 'false'
os.environ['MNEMOS_ECHOFRAME_LLM_FACING_REQUIRE_VALIDATION'] = 'true'
os.environ['MNEMOS_ECHOFRAME_LLM_FACING_ALLOW_HIGH_RISK'] = 'false'

def run():
    # Clear output directory
    import glob
    for f in glob.glob("runtime/echoframe_pilot/shadow_event_*.json"):
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
    TOTAL_CALLS = 2000
    print(f"Running {TOTAL_CALLS} mock searches to simulate full 0.05 sample rate workload...")
    
    for i in tqdm(range(TOTAL_CALLS)):
        # simulate some variance
        if i % 100 == 0:
            query = "Delete the user data for account 12345." # high-risk
        else:
            query = "What is the policy on data retention?"
            
        session_id = f"PHASE6_SIM_{uuid.uuid4().hex[:8]}"
        turn_id = f"T_{i}"
        
        EchoFrameShadowAdapter.observe_search(query, baseline_payload, session_id, turn_id)

if __name__ == "__main__":
    run()
