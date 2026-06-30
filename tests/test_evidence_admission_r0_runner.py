from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_evidence_admission_r0_comparison as runner


def _load_repo_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_runner_requires_explicit_execution_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--pack-path", str(tmp_path / "pack.json"), "--result-path", str(tmp_path / "out.json")])


def test_http_identity_fails_closed_when_revision_unverified(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_get_json", lambda url, timeout_s: {"status": "ok"})

    identity = runner.establish_http_service_identity("http://localhost:8700", timeout_s=1.0)

    assert identity["verified"] is False
    assert identity["status_code"] == "SERVICE_REVISION_UNVERIFIED"
    assert identity["formal_claim_permitted"] is False


def test_http_identity_accepts_service_revision_from_health(monkeypatch) -> None:
    def fake_get_json(url: str, timeout_s: float) -> dict:
        if url.endswith("/health"):
            return {"status": "ok", "service_revision": {"git_revision": "abc123"}}
        return {}

    monkeypatch.setattr(runner, "_get_json", fake_get_json)

    identity = runner.establish_http_service_identity("http://localhost:8700", timeout_s=1.0)

    assert identity["verified"] is True
    assert identity["identity"] == "git:abc123"
    assert identity["formal_claim_permitted"] is True


def test_collection_snapshot_uses_http_qdrant_stats() -> None:
    identity = {
        "observations": {
            "stats": {
                "stats": {
                    "retrieval": {
                        "tiers": {
                            "qdrant": {
                                "collection": "mnemos_engrams",
                                "document_count": 42,
                            }
                        }
                    }
                }
            }
        }
    }

    assert runner.collection_snapshot_from_http_identity(identity) == "mnemos_engrams:42"


def test_collection_snapshot_preserves_zero_document_count() -> None:
    identity = {
        "observations": {
            "stats": {
                "stats": {
                    "retrieval": {
                        "tiers": {
                            "qdrant": {
                                "collection": "mnemos_engrams",
                                "document_count": 0,
                            }
                        }
                    }
                }
            }
        }
    }

    assert runner.collection_snapshot_from_http_identity(identity) == "mnemos_engrams:0"


def test_manifest_contains_required_execution_fields(tmp_path: Path, monkeypatch) -> None:
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(
        json.dumps(
            {
                "pack_id": "r0-dev",
                "pack_type": "development",
                "queries": [{"query_id": "q1", "query": "What is GateMem G4 status?"}],
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "result.json"

    monkeypatch.setattr(runner, "git_head", lambda: "runner123")
    monkeypatch.setattr(runner, "collection_snapshot_for_direct_runtime", lambda: "direct:seed")
    monkeypatch.setattr(
        runner,
        "run_single_direct_runtime_query",
        lambda query, top_k, request_flag_state, global_gate_state, filters=None: {
            "normal_retrieval": {"meta": {"retrieval_mode": "semantic"}, "top_results": []},
            "pre_retrieval_recommendation": {"status": "recommended", "reason_codes": ["ADMISSION_X"]},
            "post_retrieval_sufficiency": {"sufficiency": "INSUFFICIENT_MORE_EVIDENCE_NEEDED", "reason_codes": ["SUFFICIENCY_Y"]},
            "raw_shadow_block_redacted": {"status": "recommended"},
        },
    )

    runner.main(
        [
            "--execution-mode",
            "direct_runtime",
            "--pack-path",
            str(pack_path),
            "--result-path",
            str(result_path),
            "--request-flag-state",
            "true",
            "--global-gate-state",
            "enabled",
        ]
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = payload["run_manifest"]
    assert manifest["execution_mode"] == "direct_runtime"
    assert manifest["service_base_url"] is None
    assert manifest["service_revision_or_image_identity"] == "direct_runtime:runner123"
    assert manifest["runner_commit"] == "runner123"
    assert manifest["collection_or_corpus_snapshot"] == "direct:seed"
    assert manifest["request_flag_state"] == "true"
    assert manifest["global_gate_state"] == "enabled"
    assert payload["formal_claim_permitted"] is False
    assert payload["per_query_results"][0]["pre_retrieval_recommendation"]["reason_codes"] == ["ADMISSION_X"]
    assert payload["per_query_results"][0]["post_retrieval_sufficiency"]["reason_codes"] == ["SUFFICIENCY_Y"]


def test_http_mode_does_not_fall_back_when_identity_unverified(tmp_path: Path, monkeypatch) -> None:
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(
        json.dumps({"pack_id": "r0-formal", "pack_type": "formal_evaluation", "queries": []}),
        encoding="utf-8",
    )
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(
        runner,
        "establish_http_service_identity",
        lambda base_url, timeout_s: {
            "verified": False,
            "identity": None,
            "status_code": "SERVICE_REVISION_UNVERIFIED",
            "formal_claim_permitted": False,
        },
    )

    exit_code = runner.main(
        [
            "--execution-mode",
            "http_service",
            "--service-base-url",
            "http://localhost:8700",
            "--pack-path",
            str(pack_path),
            "--result-path",
            str(result_path),
        ]
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["run_manifest"]["execution_mode"] == "http_service"
    assert payload["run_manifest"]["service_revision_or_image_identity"] == "SERVICE_REVISION_UNVERIFIED"
    assert payload["formal_claim_permitted"] is False
    assert payload["per_query_results"] == []


def test_shadow_block_is_split_and_redacted() -> None:
    response = {
        "results": [{"rank": 1, "score": 0.9, "engram": {"id": "e1", "source": "docs/a.md", "metadata": {"source_uri": "docs/a.md"}}}],
        "meta": {
            "retrieval_mode": "semantic",
            "retrieval_fingerprint": {"retrieval_profile": "semantic|balanced"},
            "evidence_admission_shadow": {
                "status": "recommended",
                "recommended_route": "SEMANTIC_RETRIEVAL",
                "candidate_budget": 8,
                "context_token_budget": 1200,
                "expansion_budget": 0,
                "latency_budget_ms": None,
                "stop_condition": "minimum_evidence_satisfied",
                "reason_codes": ["ADMISSION_STANDARD_LOOKUP_DEFAULT"],
                "sufficiency": "SUFFICIENT",
                "sufficiency_reason_codes": ["SUFFICIENCY_LINEAGE_COMPLETE_CURRENT_STATE_PRESENT"],
                "input_snapshot": "sha256:abc",
                "latency_ms": 0.1,
                "non_authoritative": True,
            },
        },
    }

    record = runner.build_query_record("q1", "What is GateMem G4 status?", response)

    assert "GateMem" not in json.dumps(record)
    assert record["query"]["redacted"] is True
    assert record["pre_retrieval_recommendation"]["reason_codes"] == ["ADMISSION_STANDARD_LOOKUP_DEFAULT"]
    assert "sufficiency" not in record["pre_retrieval_recommendation"]
    assert record["post_retrieval_sufficiency"]["sufficiency"] == "SUFFICIENT"
    assert "recommended_route" not in record["post_retrieval_sufficiency"]


def test_frozen_r0_packs_are_separate_and_mode_scoped() -> None:
    development = _load_repo_json("docs/experiments/evidence_admission_r0_development_pack.json")
    formal = _load_repo_json("docs/experiments/evidence_admission_r0_formal_evaluation_pack.json")
    fresh = _load_repo_json("docs/experiments/evidence_admission_r0_fresh_verification_pack.json")

    assert development["pack_type"] == "development"
    assert development["intended_execution_mode"] == "direct_runtime"
    assert formal["pack_type"] == "formal_evaluation"
    assert formal["intended_execution_mode"] == "http_service"
    assert fresh["pack_type"] == "fresh_verification"
    assert fresh["intended_execution_mode"] == "http_service"
    assert development["pack_id"] != formal["pack_id"] != fresh["pack_id"]
    assert "Do not aggregate direct_runtime and http_service results into one metric." in formal["rules"]
    assert all("query_id" in item and "query" in item for item in formal["queries"])


def test_runner_honors_per_query_flag_and_gate_state(tmp_path: Path, monkeypatch) -> None:
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(
        json.dumps(
            {
                "pack_id": "r0-direct",
                "pack_type": "development",
                "queries": [
                    {
                        "query_id": "off",
                        "query": "What is GateMem G4 status?",
                        "request_flag_state": "false",
                        "global_gate_state": "enabled",
                    },
                    {
                        "query_id": "disabled",
                        "query": "What is GateMem G4 status?",
                        "request_flag_state": "true",
                        "global_gate_state": "disabled",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "result.json"
    observed = []

    monkeypatch.setattr(runner, "git_head", lambda: "runner123")
    monkeypatch.setattr(runner, "collection_snapshot_for_direct_runtime", lambda: "direct:seed")

    def fake_run(query, top_k, request_flag_state, global_gate_state, filters=None):
        observed.append((request_flag_state, global_gate_state))
        return {
            "normal_retrieval": {"meta": {}, "top_results": []},
            "pre_retrieval_recommendation": None,
            "post_retrieval_sufficiency": None,
            "raw_shadow_block_redacted": {},
        }

    monkeypatch.setattr(runner, "run_single_direct_runtime_query", fake_run)

    assert runner.main(
        [
            "--execution-mode",
            "direct_runtime",
            "--pack-path",
            str(pack_path),
            "--result-path",
            str(result_path),
        ]
    ) == 0

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert observed == [("false", "enabled"), ("true", "disabled")]
    assert payload["per_query_results"][0]["request_flag_state"] == "false"
    assert payload["per_query_results"][1]["global_gate_state"] == "disabled"
