import os
import json
import pytest

from tools.smc_7b_real_shadow_eval_runner import run_shadow_evaluation, OUTPUT_DIR

def test_smc7b_runner_executes():
    run_shadow_evaluation()
    
    assert os.path.exists(os.path.join(OUTPUT_DIR, "smc_7b_real_shadow_eval_metrics.json"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "smc_7b_real_shadow_eval_report.md"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "baseline_answer_examples.json"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "shadow_answer_examples.json"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "derived_fact_usage_examples.json"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "masked_fact_examples.json"))
    assert os.path.exists(os.path.join(OUTPUT_DIR, "llm_judge_review_labels.json"))

def test_shadow_metrics_hard_gates():
    with open(os.path.join(OUTPUT_DIR, "smc_7b_real_shadow_eval_metrics.json"), "r") as f:
        metrics = json.load(f)
        
    assert metrics["contradiction_delta"] == 0.0
    assert metrics["citation_integrity_rate"] >= 95.0
    assert metrics["source_traceability_rate"] == 100.0
    assert metrics["governance_masking_rate"] == 100.0
    assert metrics["default_retrieval_leakage_count"] == 0
    assert metrics["source_mutation_count"] == 0

def test_shadow_answers_generated():
    with open(os.path.join(OUTPUT_DIR, "shadow_answer_examples.json"), "r") as f:
        answers = json.load(f)[0]
        
    assert "baseline_answer" in answers
    assert "shadow_answer" in answers
    assert "source: e_2" in answers["shadow_answer"].lower()

def test_masking_is_enforced_read_only():
    with open(os.path.join(OUTPUT_DIR, "masked_fact_examples.json"), "r") as f:
        masked = json.load(f)
        
    assert len(masked) == 1
    assert masked[0]["fact_id"] == "f_3"
    assert "vetoed" in masked[0]["reason"]
