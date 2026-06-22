"""Build or verify frozen Phase 5 human-review packets.

Offline design utility only. It reads the frozen R1 fixture and S1 replay
logic, writes reviewer packets plus a coordinator-only manifest, and has no
MNEMOS runtime, route, retrieval, governance, promotion, or memory-write path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prototype.session_context_assembler.corpus import load_validated_corpus  # noqa: E402
from prototype.session_context_assembler.replay import run_replay  # noqa: E402

CORPUS_PATH = REPO_ROOT / "benchmarks" / "truthsets" / "session_context_assembler_r1.json"
CORPUS_MANIFEST_PATH = REPO_ROOT / "benchmarks" / "truthsets" / "session_context_assembler_r1.manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "review_packets" / "session_context_assembler_phase_5"
PROTOCOL_PATH = REPO_ROOT / "docs" / "session_context_assembler_phase_5_human_review_protocol.md"
FORM_PATH = REPO_ROOT / "docs" / "session_context_assembler_phase_5_review_form.md"
SELECTOR_PATH = REPO_ROOT / "prototype" / "session_context_assembler" / "selector_s1.py"

STUDY_ID = "SCA-PHASE5-R1-S1"
PACKET_SCHEMA = "sca_phase5_review_packet_v1"
MANIFEST_SCHEMA = "sca_phase5_coordinator_manifest_v1"
DEFAULT_SEED = 20260622

CONDITIONS = (
    "A_full_history",
    "B_sliding_window",
    "C1_selector_s1_mandatory_preservation",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: Dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _condition_order(task_index: int, seed: int) -> List[str]:
    """Seeded Latin rotation: deterministic and position-balanced to ±1."""
    base = list(CONDITIONS)
    random.Random(seed).shuffle(base)
    block = (task_index - 1) // len(base)
    if block % 2:
        base = list(reversed(base))
    offset = (task_index - 1) % len(base)
    return base[offset:] + base[:offset]


def _turn_view(turn: Dict) -> Dict:
    view = {
        "turn_id": turn["turn_id"],
        "speaker": turn["speaker"],
        "content": turn["content"],
    }
    if turn.get("linked_source_ids"):
        view["source_links"] = sorted(turn["linked_source_ids"])
    return view


def _ordinary_package(record: Dict, turns_by_id: Dict[str, Dict]) -> Dict:
    return {
        "presentation": "conversation_context",
        "artifacts": [
            {
                "artifact_type": "conversation_turn",
                **_turn_view(turns_by_id[turn_id]),
            }
            for turn_id in record["selected_turn_ids"]
        ],
    }


def _synthetic_package(record: Dict, turns_by_id: Dict[str, Dict]) -> Dict:
    artifacts = []
    for label in record["synthetic_context_labels"]:
        artifacts.append(
            {
                "artifact_type": "synthetic_context",
                "label": "synthetic_context",
                "synthetic_context": True,
                "non_authoritative": bool(label["non_authoritative"]),
                "non_promotable": bool(label["non_promotable"]),
                "parent_engram_ids": list(label["parent_engram_ids"]),
                "parent_source_ids": list(label["parent_source_ids"]),
                "session_turns": [
                    _turn_view(turns_by_id[turn_id])
                    for turn_id in label["parent_turn_ids"]
                ],
            }
        )
    package = {"presentation": "artifact_context", "artifacts": artifacts}
    if record["context_budget_insufficient"]:
        package["warning"] = {
            "context_budget_insufficient": True,
            "omitted_required_artifact_types": list(
                record["omitted_required_artifact_types"]
            ),
            "selection_abstention_reason": (
                "One or more mandatory eligible artifact types could not fit "
                "inside the context budget; lower-priority semantic fill was withheld."
            ),
        }
    return package


def build_packet_documents(seed: int = DEFAULT_SEED) -> tuple[List[Dict], Dict]:
    corpus = load_validated_corpus(CORPUS_PATH, CORPUS_MANIFEST_PATH)
    corpus_manifest = json.loads(CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))
    records = run_replay(
        corpus, corpus_manifest["file_sha256"], seed=7, include_s1=True
    )
    records_by_case_condition = {
        (record["case_id"], record["condition"]): record for record in records
    }

    packets: List[Dict] = []
    coordinator_tasks = []
    for index, case in enumerate(corpus["cases"], start=1):
        task_code = f"TASK-{index:03d}"
        turns_by_id = {
            turn["turn_id"]: turn for turn in case["conversation_history"]
            if turn.get("eligible", True)
        }
        condition_order = _condition_order(index, seed)
        packages = []
        condition_key = {}
        for package_index, condition in enumerate(condition_order, start=1):
            condition_code = f"PACKAGE-{package_index}"
            record = records_by_case_condition[(case["id"], condition)]
            package = (
                _synthetic_package(record, turns_by_id)
                if condition == "C1_selector_s1_mandatory_preservation"
                else _ordinary_package(record, turns_by_id)
            )
            packages.append({"condition_code": condition_code, **package})
            condition_key[condition_code] = condition

        packet = {
            "schema": PACKET_SCHEMA,
            "study_id": STUDY_ID,
            "task_code": task_code,
            "task_prompt": case["current_task"],
            "review_instruction": (
                "Review each package independently before making the comparative rating. "
                "Package codes do not identify their construction condition."
            ),
            "packages": packages,
        }
        packets.append(packet)
        coordinator_tasks.append(
            {
                "task_code": task_code,
                "internal_case_key": case["id"],
                "condition_key": condition_key,
            }
        )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "frozen": True,
        "design_only_review_not_run": True,
        "packet_seed": seed,
        "selector_seed": 7,
        "corpus_manifest_sha256": corpus_manifest["file_sha256"],
        "packet_count": len(packets),
        "packages_per_packet": 3,
        "coordinator_only_condition_key": coordinator_tasks,
        "design_artifact_sha256": {
            "protocol": _sha256_file(PROTOCOL_PATH),
            "review_form": _sha256_file(FORM_PATH),
            "selector_s1": _sha256_file(SELECTOR_PATH),
            "packet_builder": _sha256_file(Path(__file__).resolve()),
        },
    }
    return packets, manifest


def _materialize(output_dir: Path, seed: int, overwrite: bool) -> Dict:
    packets, manifest = build_packet_documents(seed)
    packets_dir = output_dir / "packets"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"review packet directory is non-empty: {output_dir}; use --verify "
            "or explicitly --overwrite during authorized design revision"
        )
    packets_dir.mkdir(parents=True, exist_ok=True)
    packet_entries = []
    for packet in packets:
        relative = Path("packets") / f"{packet['task_code'].lower()}.json"
        data = _json_bytes(packet)
        (output_dir / relative).write_bytes(data)
        packet_entries.append(
            {
                "task_code": packet["task_code"],
                "path": relative.as_posix(),
                "sha256": _sha256_bytes(data),
            }
        )
    manifest["packets"] = packet_entries
    (output_dir / "coordinator_manifest.json").write_bytes(_json_bytes(manifest))
    (output_dir / "REVIEWER_DISTRIBUTION_NOTICE.txt").write_text(
        "Distribute only files under packets/. Do not distribute "
        "coordinator_manifest.json; it contains the condition key.\n",
        encoding="utf-8",
    )
    return manifest


def _verify(output_dir: Path, seed: int) -> None:
    manifest_path = output_dir / "coordinator_manifest.json"
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    packets, expected = build_packet_documents(seed)
    expected_by_task = {packet["task_code"]: _json_bytes(packet) for packet in packets}
    for entry in recorded["packets"]:
        data = (output_dir / entry["path"]).read_bytes()
        if _sha256_bytes(data) != entry["sha256"]:
            raise ValueError(f"packet hash mismatch: {entry['path']}")
        if data != expected_by_task[entry["task_code"]]:
            raise ValueError(f"packet is not reproducible: {entry['path']}")
    for key in (
        "schema", "study_id", "packet_seed", "selector_seed",
        "corpus_manifest_sha256", "packet_count", "packages_per_packet",
        "coordinator_only_condition_key", "design_artifact_sha256",
    ):
        if recorded[key] != expected[key]:
            raise ValueError(f"manifest mismatch for {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.verify:
        _verify(output_dir, args.seed)
        print(f"Verified frozen reviewer packets: {output_dir}")
    else:
        manifest = _materialize(output_dir, args.seed, args.overwrite)
        print(
            f"Built {manifest['packet_count']} condition-masked task packets at "
            f"{output_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
