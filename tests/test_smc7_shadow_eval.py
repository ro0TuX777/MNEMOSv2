import os
import json
import pytest

from tools.smc_7_shadow_eval_runner import run_shadow_evaluation, OUTPUT_DIR, setup_synthetic_env

def test_smc7_runner_executes_and_outputs_files():
    run_shadow_evaluation()
    
    assert os.path.exists(os.path.join(OUTPUT_DIR, "smc_7_shadow_eval_metrics.json"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "smc_7_shadow_eval_report.md"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "baseline_packet_examples.json"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "shadow_packet_examples.json"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "derived_fact_usage_examples.json"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "masked_fact_examples.json"))

def test_shadow_metrics_pass_gates():
    with open(os.path.join(OUTPUT_DIR, "smc_7_shadow_eval_metrics.json"), "r") as f:
        metrics = json.load(f)
        
    assert metrics["contradiction_delta"] == 0.0
    assert metrics["citation_integrity_rate"] >= 95.0
    assert metrics["source_traceability_rate"] == 100.0
    assert metrics["governance_masking_rate"] == 100.0

def test_shadow_packet_structure():
    with open(os.path.join(OUTPUT_DIR, "shadow_packet_examples.json"), "r") as f:
        shadow_pkt = json.load(f)[0]
        
    assert "derived_facts" in shadow_pkt
    # Fact 1 and 2 are valid
    assert len(shadow_pkt["derived_facts"]) == 2
    for fact_str in shadow_pkt["derived_facts"]:
        assert "[Derived FactNode]" in fact_str
        assert "source_engram_id=" in fact_str
        assert "passage_node_id=" in fact_str
        assert "promotion_receipt_id=" in fact_str

def test_masking_is_enforced():
    with open(os.path.join(OUTPUT_DIR, "masked_fact_examples.json"), "r") as f:
        masked = json.load(f)
        
    assert len(masked) == 2
    f_ids = [m["fact_id"] for m in masked]
    assert "f_3" in f_ids # Suppressed
    assert "f_4" in f_ids # Downgraded
