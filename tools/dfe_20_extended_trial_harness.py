import sys
import json
import uuid
import hashlib
from unittest.mock import patch, MagicMock
from mnemos.engram.model import Engram
from service.app import app, _runtime

def run_dfe_20_harness():
    print("=========================================================")
    print("DFE-20 EXTENDED OPERATOR TRIAL VALIDATION HARNESS")
    print("=========================================================\n")

    app.config['TESTING'] = True

    with patch('mnemos.retrieval.qdrant_tier.QdrantTier._initialize'):
        _runtime.initialize()

    with app.test_client() as client:
        # We need to mock _router.search_derived so it returns fake derived results
        # just so we can test the API integration.
        fake_router_response = {
            "derived_results": [
                {
                    "content": "A derived fact for testing.",
                    "authority_label": "MNEMOS_DERIVED_FACT",
                    "rendered_support_excerpt": "This is a source excerpt.",
                    "source_document": "source.pdf",
                    "source_engram_id": "engram_123",
                    "selection_path": "STANDARD",
                    "governance": {
                        "veto_pass": True,
                        "conflict_status": "none"
                    }
                },
                {
                    "content": "An invalid fact missing an excerpt.",
                    "authority_label": "MNEMOS_DERIVED_FACT",
                    # MISSING rendered_support_excerpt
                    "source_document": "source.pdf",
                    "source_engram_id": "engram_456",
                    "selection_path": "STANDARD"
                }
            ],
            "derived_lane_meta": {
                "candidate_telemetry": [{"metric": "val"}]
            }
        }
        
        with patch.object(_runtime._router, 'search_derived', return_value=fake_router_response) as mock_search_derived:
            
            # --- GATE 1: DEFAULT RETRIEVAL UNCHANGED ---
            print("[1] Verifying Default Retrieval Is Unchanged...")
            resp = client.post("/api/v1/query", json={"query": "test query"})
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert "derived_results" not in data, "Default retrieval leaked derived facts!"
            assert "derived_lane_meta" not in data, "Default retrieval leaked derived lane meta!"
            print("    PASS: Default retrieval returns 0 derived facts.")
            
            # --- GATE 2: LEGACY EVALUATION_MODE REJECTED ---
            print("\n[2] Verifying Legacy evaluation_mode=true Is Rejected...")
            resp = client.post("/api/v1/query", json={"query": "test query", "evaluation_mode": True})
            assert resp.status_code == 400
            assert "not supported on production routes" in json.loads(resp.data)["error"]
            print("    PASS: Legacy parameter rejected.")
            
            # Setup configuration for the new tests
            with patch('service.app.get_config') as mock_get_config:
                mock_conf = MagicMock()
                mock_conf.token = ""
                mock_conf.derived_enabled = True
                mock_conf.derived_whitelist = ["DFE_OPERATOR_01"]
                mock_get_config.return_value = mock_conf
                
                # --- GATE 3: FEATURE FLAG WITH NO ALLOWLIST ---
                print("\n[3] Verifying Feature Flag Requires Allowlist...")
                headers = {"X-Client-Id": "UNAUTHORIZED_USER"}
                resp = client.post("/api/v1/query", headers=headers, json={"query": "test query", "enable_derived_facts": True})
                assert resp.status_code == 403
                assert "client_not_authorized" in json.loads(resp.data)["error"]
                print("    PASS: Unauthorized user rejected.")
                
                # --- GATE 4: KILL SWITCH ---
                print("\n[4] Verifying Kill-Switch...")
                mock_conf.derived_enabled = False
                headers = {"X-Client-Id": "DFE_OPERATOR_01"}
                resp = client.post("/api/v1/query", headers=headers, json={"query": "test query", "enable_derived_facts": True})
                assert resp.status_code == 503
                assert "derived_lane_disabled" in json.loads(resp.data)["error"]
                print("    PASS: Kill-switch successfully blocks access.")
                
                # --- GATE 5: SUCCESSFUL TRIAL REQUEST WITH FILTERING ---
                print("\n[5] Verifying Successful Execution and Schema Filtering...")
                mock_conf.derived_enabled = True
                headers = {"X-Client-Id": "DFE_OPERATOR_01"}
                resp = client.post("/api/v1/query", headers=headers, json={"query": "test query", "enable_derived_facts": True})
                assert resp.status_code == 200
                data = json.loads(resp.data)
                
                # Verify exact structure
                assert "derived_lane_meta" in data, "Missing derived_lane_meta block"
                assert "derived_results" not in data, "derived_results should not be at root level"
                
                derived_meta = data["derived_lane_meta"]
                assert "candidate_telemetry" in derived_meta
                
                # Verify filtering (1 valid, 1 invalid missing excerpt)
                facts = derived_meta["derived_results"]
                assert len(facts) == 1, f"Expected exactly 1 validated fact, got {len(facts)}"
                assert facts[0]["content"] == "A derived fact for testing."
                print("    PASS: Derived facts correctly quarantined and filtered.")
                
    print("\n=========================================================")
    print("ALL DFE-20 TECHNICAL SAFETY GATES PASSED")
    print("=========================================================")

if __name__ == "__main__":
    run_dfe_20_harness()
