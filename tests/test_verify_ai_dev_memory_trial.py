import json

from tools.verify_ai_dev_memory_trial import verify_trial_folder


def _write(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_legacy_no_memory_trial_is_valid_but_not_e1_ready(tmp_path):
    root = tmp_path / "no_memory"
    root.mkdir()
    _write(
        root / "run_manifest.json",
        json.dumps(
            {
                "run_label": "AI_DEV_MEMORY_TRIAL_NO_MEMORY_CONTROL",
                "memory_condition": "no_memory",
                "agent": "unknown",
                "model": "unknown",
                "started_at": "2026-06-27T10:00:00Z",
                "app_task": "Local Issue Tracker",
                "token_counts_available": False,
                "claim_boundary": {
                    "local_development_evidence_only": True,
                    "general_memory_claim": False,
                },
                "completed_at": "2026-06-27T10:10:00Z",
                "final_status": "completed",
                "estimated_input_tokens": 10,
                "estimated_output_tokens": 20,
            }
        ),
    )
    for name in (
        "agent_route_log.jsonl",
        "repo_activity.jsonl",
        "user_interventions.jsonl",
        "wrong_turns.jsonl",
        "test_runs.jsonl",
    ):
        _write(root / name, json.dumps({"timestamp": "2026-06-27T10:00:01Z"}) + "\n")
    _write(root / "final_report.md", "# done\n")
    _write(root / "blockers.md", "")

    result = verify_trial_folder(root, condition="no_memory")
    assert result["valid"] is True
    assert result["e1_ready"] is False


def test_e1_mnemos_trial_is_e1_ready(tmp_path):
    root = tmp_path / "mnemos_enabled"
    root.mkdir()
    _write(
        root / "run_manifest.json",
        json.dumps(
            {
                "trial_id": "trial-001",
                "run_label": "AI_DEV_MEMORY_TRIAL_MNEMOS_ENABLED",
                "memory_condition": "mnemos_enabled",
                "task_id": "local_issue_tracker_v1",
                "task_spec_hash": "abc123",
                "agent": "unknown",
                "model": "unknown",
                "client_version": "client-1",
                "started_at": "2026-06-27T10:00:00Z",
                "app_task": "Local Issue Tracker",
                "repo_root": "D:/project",
                "initial_repo_commit_or_hash": "deadbeef",
                "tool_configuration": {
                    "shell": "powershell",
                    "test_command": "npm test",
                    "build_command": "npm run build",
                },
                "mnemos_base_url": "http://localhost:8700",
                "mcp_server_name": "mnemos",
                "mnemos_service_revision": "svc-1",
                "mcp_revision": "mcp-1",
                "collection_name": "mnemos-test",
                "seed_snapshot": "seed-1",
                "configured_retrieval_profile": "hybrid",
                "execution_path": "mcp",
                "cache_state_at_start": "cold",
                "cache_policy_version": "cache-v1",
                "token_counts_available": False,
                "token_accounting_method": "estimated",
                "claim_boundary": {
                    "local_development_evidence_only": True,
                    "general_memory_claim": False,
                },
                "completed_at": "2026-06-27T10:10:00Z",
                "final_status": "completed",
                "estimated_input_tokens": 10,
                "estimated_output_tokens": 20,
                "acceptance_test_command": "npm test",
                "acceptance_test_result": "pass",
                "cache_state_at_end": "warm",
            }
        ),
    )
    _write(
        root / "memory_calls.jsonl",
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "timestamp": "2026-06-27T10:00:01Z",
                    "tool": "health_check",
                    "purpose": "startup",
                    "query_or_summary": "health",
                    "query_text": "health",
                    "retrieval_fingerprint": "not_available",
                    "execution_path": "mcp",
                    "cache_state_observed": "unknown",
                    "cache_hit": False,
                    "configured_retrieval_profile": "hybrid",
                    "duplicate_suppression_count": 0,
                    "result_count": 1,
                    "returned_source_ids": [],
                    "returned_source_labels": [],
                    "returned_source_types": [],
                    "returned_scores": [],
                    "used_result_ids": [],
                    "rejected_result_ids": [],
                    "abstention_reason": None,
                    "verified_against_primary_evidence": True,
                    "used_in_next_action": True,
                    "next_action_summary": "continue startup",
                    "helpfulness": "useful",
                    "notes": "",
                },
                {
                    "timestamp": "2026-06-27T10:00:02Z",
                    "tool": "get_capabilities",
                    "purpose": "startup",
                    "query_or_summary": "capabilities",
                    "query_text": "capabilities",
                    "retrieval_fingerprint": "not_available",
                    "execution_path": "mcp",
                    "cache_state_observed": "unknown",
                    "cache_hit": False,
                    "configured_retrieval_profile": "hybrid",
                    "duplicate_suppression_count": 0,
                    "result_count": 1,
                    "returned_source_ids": [],
                    "returned_source_labels": [],
                    "returned_source_types": [],
                    "returned_scores": [],
                    "used_result_ids": [],
                    "rejected_result_ids": [],
                    "abstention_reason": None,
                    "verified_against_primary_evidence": True,
                    "used_in_next_action": True,
                    "next_action_summary": "continue startup",
                    "helpfulness": "useful",
                    "notes": "",
                },
                {
                    "timestamp": "2026-06-27T10:01:00Z",
                    "tool": "search_memory",
                    "purpose": "orientation",
                    "query_or_summary": "find tracker context",
                    "query_text": "issue tracker app context",
                    "retrieval_fingerprint": "fp-1",
                    "execution_path": "mcp",
                    "cache_state_observed": "cache_miss",
                    "cache_hit": False,
                    "configured_retrieval_profile": "hybrid",
                    "duplicate_suppression_count": 1,
                    "result_count": 2,
                    "returned_source_ids": ["src-1", "src-2"],
                    "returned_source_labels": ["doc1", "doc2"],
                    "returned_source_types": ["summary", "full_document"],
                    "returned_scores": [0.9, 0.7],
                    "used_result_ids": ["src-1"],
                    "rejected_result_ids": ["src-2"],
                    "abstention_reason": None,
                    "verified_against_primary_evidence": True,
                    "used_in_next_action": True,
                    "next_action_summary": "open local files",
                    "helpfulness": "useful",
                    "notes": "",
                },
                {
                    "timestamp": "2026-06-27T10:02:00Z",
                    "tool": "record_decision",
                    "purpose": "decision log",
                    "query_or_summary": "record storage choice",
                    "query_text": "record storage choice",
                    "retrieval_fingerprint": "not_applicable",
                    "execution_path": "mcp",
                    "cache_state_observed": "not_applicable",
                    "cache_hit": False,
                    "configured_retrieval_profile": "hybrid",
                    "duplicate_suppression_count": 0,
                    "result_count": 0,
                    "returned_source_ids": [],
                    "returned_source_labels": [],
                    "returned_source_types": [],
                    "returned_scores": [],
                    "used_result_ids": [],
                    "rejected_result_ids": [],
                    "abstention_reason": None,
                    "verified_against_primary_evidence": True,
                    "used_in_next_action": True,
                    "next_action_summary": "implement storage utility",
                    "helpfulness": "neutral",
                    "notes": "",
                },
                {
                    "timestamp": "2026-06-27T10:05:00Z",
                    "tool": "write_observation",
                    "purpose": "bug fix note",
                    "query_or_summary": "bug fix",
                    "query_text": "record bug fix",
                    "retrieval_fingerprint": "not_applicable",
                    "execution_path": "mcp",
                    "cache_state_observed": "not_applicable",
                    "cache_hit": False,
                    "configured_retrieval_profile": "hybrid",
                    "duplicate_suppression_count": 0,
                    "result_count": 0,
                    "returned_source_ids": [],
                    "returned_source_labels": [],
                    "returned_source_types": [],
                    "returned_scores": [],
                    "used_result_ids": [],
                    "rejected_result_ids": [],
                    "abstention_reason": None,
                    "verified_against_primary_evidence": True,
                    "used_in_next_action": True,
                    "next_action_summary": "rerun tests",
                    "helpfulness": "neutral",
                    "notes": "",
                },
                {
                    "timestamp": "2026-06-27T10:09:00Z",
                    "tool": "summarize_session_handoff",
                    "purpose": "handoff",
                    "query_or_summary": "handoff summary",
                    "query_text": "handoff summary",
                    "retrieval_fingerprint": "not_applicable",
                    "execution_path": "mcp",
                    "cache_state_observed": "not_applicable",
                    "cache_hit": False,
                    "configured_retrieval_profile": "hybrid",
                    "duplicate_suppression_count": 0,
                    "result_count": 0,
                    "returned_source_ids": [],
                    "returned_source_labels": [],
                    "returned_source_types": [],
                    "returned_scores": [],
                    "used_result_ids": [],
                    "rejected_result_ids": [],
                    "abstention_reason": None,
                    "verified_against_primary_evidence": True,
                    "used_in_next_action": True,
                    "next_action_summary": "write final report",
                    "helpfulness": "neutral",
                    "notes": "",
                },
            ]
        )
        + "\n",
    )
    for name in (
        "agent_route_log.jsonl",
        "repo_activity.jsonl",
        "user_interventions.jsonl",
        "wrong_turns.jsonl",
        "test_runs.jsonl",
    ):
        _write(root / name, json.dumps({"timestamp": "2026-06-27T10:00:01Z"}) + "\n")
    _write(root / "final_report.md", "# done\n")
    _write(root / "blockers.md", "")

    result = verify_trial_folder(root, condition="mnemos_enabled")
    assert result["valid"] is True
    assert result["e1_ready"] is True
