"""Run the fixed-model A/B/C1 answer-fidelity surrogate on frozen R2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.session_context_assembler.corpus import load_validated_corpus  # noqa: E402
from prototype.session_context_assembler.replay import (  # noqa: E402
    run_condition_a,
    run_condition_b,
    run_condition_c1,
)

CORPUS = ROOT / "benchmarks" / "truthsets" / "session_context_assembler_r2.json"
MANIFEST = ROOT / "benchmarks" / "truthsets" / "session_context_assembler_r2.manifest.json"
RESULT_JSON = ROOT / "benchmarks" / "results" / "session_context_assembler_model_fidelity.json"
RESULT_MD = ROOT / "benchmarks" / "results" / "session_context_assembler_model_fidelity.md"
DEFAULT_MODEL = "hf.co/WSDW/Qwen2.5-7B-Instruct-Q4_K_M-GGUF:Q4_K_M"
DEFAULT_MODEL_DIGEST = "sha256:3105a0828a9d92d24ce55b75cc2bee9fbadaa60de5605e8b440bb847eef7f8b0"
CONDITIONS = {
    "A": run_condition_a,
    "B": run_condition_b,
    "C1": run_condition_c1,
}


def _normalize_host(host: str) -> str:
    value = host.strip()
    if "://" not in value:
        value = "http://" + value
    return value.replace("http://0.0.0.0", "http://127.0.0.1", 1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt(case: dict[str, Any], record: dict[str, Any]) -> str:
    turns = {turn["turn_id"]: turn for turn in case["conversation_history"]}
    context = []
    for turn_id in record["selected_turn_ids"]:
        turn = turns[turn_id]
        context.append(
            {
                "turn_id": turn_id,
                "speaker": turn["speaker"],
                "content": turn["content"],
                "decision_ids": [
                    item
                    for item in record["selected_parent_engram_ids"]
                    if item in turn["content"]
                ],
                "source_ids": turn.get("linked_source_ids", []),
            }
        )
    payload = {
        "task": case["current_task"],
        "context": context,
        "available_decision_ids": record["selected_parent_engram_ids"],
        "available_source_ids": record["selected_source_ids"],
        "context_budget_insufficient": record.get("context_budget_insufficient", False),
        "omitted_required_artifact_types": record.get("omitted_required_artifact_types", []),
    }
    return (
        "Answer only from the supplied context. Do not infer missing facts. "
        "If context_budget_insufficient is true, explicitly acknowledge that the answer is incomplete. "
        "If the context contains unresolved or conflicting statements, explicitly say they are unresolved. "
        "Return one JSON object with exactly these keys: answer (string), cited_turn_ids (array), "
        "cited_source_ids (array), cited_decision_ids (array), unsupported_claims (array of strings), "
        "contradiction_acknowledged (boolean), abstention_acknowledged (boolean).\nINPUT:\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def ollama_generate(prompt: str, *, model: str, host: str) -> dict[str, Any]:
    request = urllib.request.Request(
        _normalize_host(host).rstrip("/") + "/api/generate",
        data=json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "seed": 5101, "num_predict": 500},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        outer = json.loads(response.read().decode("utf-8"))
    value = json.loads(outer["response"])
    if not isinstance(value, dict):
        raise ValueError("model response is not an object")
    return value


def _valid_response(value: dict[str, Any]) -> bool:
    expected = {
        "answer", "cited_turn_ids", "cited_source_ids", "cited_decision_ids",
        "unsupported_claims", "contradiction_acknowledged", "abstention_acknowledged",
    }
    return (
        set(value) == expected
        and isinstance(value["answer"], str)
        and all(isinstance(value[name], list) for name in (
            "cited_turn_ids", "cited_source_ids", "cited_decision_ids", "unsupported_claims"
        ))
        and isinstance(value["contradiction_acknowledged"], bool)
        and isinstance(value["abstention_acknowledged"], bool)
    )


def run_evaluation(
    generate: Callable[[str], dict[str, Any]], *, model: str, model_digest: str
) -> dict[str, Any]:
    corpus = load_validated_corpus(CORPUS, MANIFEST)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for case in corpus["cases"]:
        budget = case["expected_context_budget"]
        records = {
            "A": run_condition_a(case),
            "B": run_condition_b(case, budget),
            "C1": run_condition_c1(case, manifest["file_sha256"], 7, budget),
        }
        expected = case["verification_expectations"]
        contradiction_expected = "contradiction" in case["verification_class"]
        for code, record in records.items():
            prompt = _prompt(case, record)
            if "verification_expectations" in prompt:
                raise ValueError("scoring-only field entered model prompt")
            try:
                response = generate(prompt)
                generation_completed = True
                valid = _valid_response(response)
            except Exception as exc:
                response = {"error": type(exc).__name__}
                generation_completed = False
                valid = False
            available_turns = set(record["selected_turn_ids"])
            available_sources = set(record["selected_source_ids"])
            available_decisions = set(record["selected_parent_engram_ids"])
            cited_turns = set(response.get("cited_turn_ids", [])) if valid else set()
            cited_sources = set(response.get("cited_source_ids", [])) if valid else set()
            cited_decisions = set(response.get("cited_decision_ids", [])) if valid else set()
            required_sources = set(expected["required_source_ids"])
            required_decisions = set(expected["required_decision_ids"])
            citations_valid = (
                cited_turns <= available_turns
                and cited_sources <= available_sources
                and cited_decisions <= available_decisions
            )
            source_preserved = required_sources <= cited_sources
            decision_preserved = required_decisions <= cited_decisions
            contradiction_correct = (
                bool(response.get("contradiction_acknowledged")) == contradiction_expected
            ) if valid else False
            abstention_expected = bool(record.get("context_budget_insufficient", False))
            abstention_correct = (
                bool(response.get("abstention_acknowledged")) == abstention_expected
            ) if valid else False
            unsupported_count = len(response.get("unsupported_claims", [])) if valid else 1
            grounded = all((
                valid, citations_valid, source_preserved, decision_preserved,
                contradiction_correct, abstention_correct, unsupported_count == 0,
            ))
            rows.append({
                "case_id": case["id"],
                "condition": code,
                "response_valid": valid,
                "generation_completed": generation_completed,
                "execution_error": response.get("error") if not generation_completed else None,
                "grounded_agreement": grounded,
                "source_id_preserved": source_preserved,
                "decision_id_preserved": decision_preserved,
                "citations_valid": citations_valid,
                "contradiction_handling_correct": contradiction_correct,
                "abstention_acknowledgement_correct": abstention_correct,
                "unsupported_claim_count": unsupported_count,
                "answer": response.get("answer") if valid else None,
                "cited_turn_ids": sorted(cited_turns),
                "cited_source_ids": sorted(cited_sources),
                "cited_decision_ids": sorted(cited_decisions),
            })
    aggregate = {}
    for code in CONDITIONS:
        subset = [row for row in rows if row["condition"] == code]
        aggregate[code] = {
            "count": len(subset),
            "valid_response_rate": sum(row["response_valid"] for row in subset) / len(subset),
            "grounded_agreement_rate": sum(row["grounded_agreement"] for row in subset) / len(subset),
            "source_id_preservation_rate": sum(row["source_id_preserved"] for row in subset) / len(subset),
            "decision_id_preservation_rate": sum(row["decision_id_preserved"] for row in subset) / len(subset),
            "contradiction_handling_rate": sum(row["contradiction_handling_correct"] for row in subset) / len(subset),
            "abstention_acknowledgement_rate": sum(row["abstention_acknowledgement_correct"] for row in subset) / len(subset),
            "unsupported_claim_rate": sum(row["unsupported_claim_count"] > 0 for row in subset) / len(subset),
        }
    completed = sum(row["generation_completed"] for row in rows)
    return {
        "schema": "session_context_assembler_model_fidelity_v1",
        "labels": ["MODEL_ASSISTED_SURROGATE_EVALUATION", "NOT_HUMAN_VALUE_EVIDENCE", "NOT_GENERALIZABLE", "NO_RUNTIME_INTEGRATION"],
        "model": model,
        "model_digest": model_digest,
        "temperature": 0,
        "seed": 5101,
        "corpus_sha256": _sha256(CORPUS),
        "case_count": len(corpus["cases"]),
        "conditions": ["A", "B", "C1"],
        "execution_status": "COMPLETE" if completed == len(rows) else "INCOMPLETE_MODEL_EXECUTION",
        "model_call_success_count": completed,
        "model_call_total": len(rows),
        "aggregate": aggregate,
        "records": rows,
        "claim_boundary": "Model-assisted answer-fidelity surrogate only; not human evidence, production readiness, or a generalization claim.",
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = ["# Session Context Assembler A/B/C1 Model-Assisted Fidelity", "", " ".join(f"`{x}`" for x in result["labels"]), "", f"Model: `{result['model']}` (`{result['model_digest']}`)", "", "| Condition | Grounded agreement | Source IDs | Contradictions | Abstention | Unsupported claims |", "|---|---:|---:|---:|---:|---:|"]
    for code, row in result["aggregate"].items():
        lines.append(f"| {code} | {row['grounded_agreement_rate']:.3f} | {row['source_id_preservation_rate']:.3f} | {row['contradiction_handling_rate']:.3f} | {row['abstention_acknowledgement_rate']:.3f} | {row['unsupported_claim_rate']:.3f} |")
    lines.extend(["", result["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("MNEMOS_FIDELITY_MODEL", DEFAULT_MODEL))
    parser.add_argument("--model-digest", default=os.getenv("MNEMOS_FIDELITY_MODEL_DIGEST", DEFAULT_MODEL_DIGEST))
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    args = parser.parse_args()
    result = run_evaluation(
        lambda prompt: ollama_generate(prompt, model=args.model, host=args.host),
        model=args.model,
        model_digest=args.model_digest,
    )
    RESULT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    RESULT_MD.write_text(_markdown(result), encoding="utf-8", newline="\n")
    print(f"Wrote {RESULT_JSON}")
    print(f"Wrote {RESULT_MD}")
    if result["execution_status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
