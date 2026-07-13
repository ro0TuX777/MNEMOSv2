from tools.run_session_context_assembler_model_fidelity import _normalize_host, run_evaluation


def _mock(_prompt_text):
    return {
        "answer": "Bounded synthetic answer.",
        "cited_turn_ids": [],
        "cited_source_ids": [],
        "cited_decision_ids": [],
        "unsupported_claims": [],
        "contradiction_acknowledged": False,
        "abstention_acknowledged": False,
    }


def test_model_fidelity_runner_covers_identical_a_b_c1_conditions():
    result = run_evaluation(_mock, model="mock-fixed", model_digest="mock-digest")
    assert result["case_count"] == 10
    assert set(result["aggregate"]) == {"A", "B", "C1"}
    assert all(row["count"] == 10 for row in result["aggregate"].values())
    assert len(result["records"]) == 30
    assert result["execution_status"] == "COMPLETE"


def test_scoring_only_expectations_never_enter_prompt():
    seen = []

    def capture(prompt):
        seen.append(prompt)
        return _mock(prompt)

    run_evaluation(capture, model="mock-fixed", model_digest="mock-digest")
    assert seen
    assert all("verification_expectations" not in prompt for prompt in seen)


def test_surrogate_claim_boundary_is_not_human_or_generalizable():
    result = run_evaluation(_mock, model="mock-fixed", model_digest="mock-digest")
    assert "MODEL_ASSISTED_SURROGATE_EVALUATION" in result["labels"]
    assert "NOT_HUMAN_VALUE_EVIDENCE" in result["labels"]
    assert "NOT_GENERALIZABLE" in result["labels"]


def test_ollama_host_is_normalized_for_local_client_use():
    assert _normalize_host("0.0.0.0:7777") == "http://127.0.0.1:7777"


def test_transport_failure_is_not_reported_as_a_scored_model_run():
    def fail(_prompt):
        raise OSError("offline")

    result = run_evaluation(fail, model="mock-fixed", model_digest="mock-digest")
    assert result["execution_status"] == "INCOMPLETE_MODEL_EXECUTION"
    assert result["model_call_success_count"] == 0
