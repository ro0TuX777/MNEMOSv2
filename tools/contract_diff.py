#!/usr/bin/env python3
"""Diff two MNEMOS contract files and evaluate compatibility.

Checks backward/forward compatibility for contract evolution by validating
required fields, type changes, status value changes, enum semantics,
and default value transitions.

Usage:
  python tools/contract_diff.py --old service/contract.json --new contracts/mnemos_v2.json --mode both
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class Finding:
    severity: str   # "error" | "warning"
    code: str
    message: str


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Contract must be a JSON object: {path}")
    return payload


def _major(version: str) -> int | None:
    m = re.fullmatch(r"v(\d+)", str(version).strip())
    return int(m.group(1)) if m else None


def _required_fields(payload: Dict[str, Any]) -> Dict[str, str]:
    fields = payload.get("required_fields", {})
    if not isinstance(fields, dict):
        return {}
    return {k: v for k, v in fields.items() if isinstance(k, str) and isinstance(v, str)}


def _allowed_status(payload: Dict[str, Any]) -> set[str]:
    raw = payload.get("allowed_status", [])
    return {s for s in raw if isinstance(s, str)} if isinstance(raw, list) else set()


def _field_semantics(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = payload.get("field_semantics", {})
    if not isinstance(raw, dict):
        return {}
    return {f: c for f, c in raw.items() if isinstance(f, str) and isinstance(c, dict)}


def _enum_values(config: Dict[str, Any]) -> set[str]:
    raw = config.get("enum")
    return {str(v) for v in raw} if isinstance(raw, list) else set()


def _validate_semantic_shape(payload: Dict[str, Any], label: str) -> List[Finding]:
    findings: List[Finding] = []
    required = _required_fields(payload)
    for field, cfg in _field_semantics(payload).items():
        if field not in required:
            findings.append(Finding("warning", "semantic_for_non_required_field",
                f"[{label}] field_semantics includes '{field}' which is not in required_fields"))
        enum_vals = _enum_values(cfg)
        if "enum" in cfg and not enum_vals:
            findings.append(Finding("warning", "invalid_enum_semantics",
                f"[{label}] field_semantics.{field}.enum is present but empty/invalid"))
        if "default" in cfg and enum_vals:
            default_val = str(cfg.get("default"))
            if default_val not in enum_vals:
                findings.append(Finding("warning", "default_not_in_enum",
                    f"[{label}] field_semantics.{field}.default='{default_val}' not in enum"))
    return findings


def _check_semantic_compatibility(
    *, producer: Dict[str, Any], consumer: Dict[str, Any], label: str,
) -> List[Finding]:
    findings: List[Finding] = []
    p_sem = _field_semantics(producer)
    c_sem = _field_semantics(consumer)
    for field in sorted(set(p_sem) & set(c_sem)):
        p_enum = _enum_values(p_sem[field])
        c_enum = _enum_values(c_sem[field])
        if p_enum and c_enum:
            unknown = sorted(p_enum - c_enum)
            if unknown:
                findings.append(Finding("error", "enum_not_accepted",
                    f"[{label}] field '{field}' producer enum has values not accepted by consumer: {', '.join(unknown)}"))
    return findings


def _type_compatible(producer_type: str, consumer_type: str) -> bool:
    if producer_type == consumer_type:
        return True
    if producer_type == "str" and consumer_type == "nullable_str":
        return True
    return False


def _check_producer_consumer(
    *, producer: Dict[str, Any], consumer: Dict[str, Any], label: str,
) -> Tuple[bool, List[Finding]]:
    findings: List[Finding] = []

    p_fields = _required_fields(producer)
    c_fields = _required_fields(consumer)

    for field, expected_type in c_fields.items():
        if field not in p_fields:
            findings.append(Finding("error", "missing_required_field",
                f"[{label}] producer missing required field '{field}' expected by consumer"))
            continue
        if not _type_compatible(p_fields[field], expected_type):
            findings.append(Finding("error", "type_mismatch",
                f"[{label}] field '{field}' type incompatible: producer='{p_fields[field]}', consumer='{expected_type}'"))

    p_status = _allowed_status(producer)
    c_status = _allowed_status(consumer)
    unknown_status = sorted(p_status - c_status)
    if unknown_status:
        findings.append(Finding("error", "status_not_accepted",
            f"[{label}] producer can emit statuses not accepted by consumer: {', '.join(unknown_status)}"))

    findings.extend(_check_semantic_compatibility(producer=producer, consumer=consumer, label=label))

    ok = not any(f.severity == "error" for f in findings)
    return ok, findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two contract files and evaluate compatibility")
    parser.add_argument("--old", required=True, help="Path to old contract JSON")
    parser.add_argument("--new", required=True, help="Path to new contract JSON")
    parser.add_argument("--mode", choices=["backward", "forward", "both"], default="both")
    parser.add_argument("--output-json", default="", help="Optional path to write detailed JSON report")
    args = parser.parse_args()

    old_path = Path(args.old).resolve()
    new_path = Path(args.new).resolve()
    if not old_path.exists():
        raise FileNotFoundError(f"Old contract not found: {old_path}")
    if not new_path.exists():
        raise FileNotFoundError(f"New contract not found: {new_path}")

    old = _load_json(old_path)
    new = _load_json(new_path)

    findings: List[Finding] = []
    findings.extend(_validate_semantic_shape(old, "old"))
    findings.extend(_validate_semantic_shape(new, "new"))

    # Detect metadata changes
    old_name = str(old.get("service_name", ""))
    new_name = str(new.get("service_name", ""))
    if old_name and new_name and old_name != new_name:
        findings.append(Finding("warning", "service_name_changed",
            f"service_name changed: '{old_name}' -> '{new_name}'"))

    old_endpoint = str(old.get("endpoint", ""))
    new_endpoint = str(new.get("endpoint", ""))
    if old_endpoint and new_endpoint and old_endpoint != new_endpoint:
        findings.append(Finding("warning", "endpoint_changed",
            f"endpoint changed: '{old_endpoint}' -> '{new_endpoint}'"))

    old_ver = str(old.get("contract_version", ""))
    new_ver = str(new.get("contract_version", ""))
    old_major = _major(old_ver)
    new_major = _major(new_ver)

    backward_ok = None
    forward_ok = None

    if args.mode in {"backward", "both"}:
        backward_ok, back_findings = _check_producer_consumer(producer=new, consumer=old, label="backward")
        findings.extend(back_findings)

    if args.mode in {"forward", "both"}:
        forward_ok, fwd_findings = _check_producer_consumer(producer=old, consumer=new, label="forward")
        findings.extend(fwd_findings)

    has_breaking = any(f.severity == "error" for f in findings)

    # Default value change warnings
    old_sem = _field_semantics(old)
    new_sem = _field_semantics(new)
    for field in sorted(set(old_sem) & set(new_sem)):
        if "default" in old_sem[field] and "default" in new_sem[field]:
            old_def = old_sem[field].get("default")
            new_def = new_sem[field].get("default")
            if old_def != new_def:
                findings.append(Finding("warning", "default_changed",
                    f"field_semantics.{field}.default changed: {old_def!r} -> {new_def!r}"))

    # Major version bump advisories
    if old_major is not None and new_major is not None:
        if has_breaking and new_major <= old_major:
            findings.append(Finding("warning", "recommended_major_bump",
                f"breaking changes detected but version did not bump major ({old_ver} -> {new_ver}); consider incrementing major"))
        if not has_breaking and new_major > old_major:
            findings.append(Finding("warning", "potential_unnecessary_major_bump",
                f"no breaking changes detected but major version increased ({old_ver} -> {new_ver})"))

    # Report
    print(f"old: {old_path}")
    print(f"new: {new_path}")
    print(f"mode: {args.mode}")
    if backward_ok is not None:
        print(f"backward_compatible: {'yes' if backward_ok else 'no'}")
    if forward_ok is not None:
        print(f"forward_compatible: {'yes' if forward_ok else 'no'}")

    if findings:
        print("\nfindings:")
        for item in findings:
            print(f"- [{item.severity}] {item.code}: {item.message}")
    else:
        print("\nfindings:\n- none")

    report = {
        "old": str(old_path),
        "new": str(new_path),
        "mode": args.mode,
        "backward_compatible": backward_ok,
        "forward_compatible": forward_ok,
        "findings": [
            {"severity": f.severity, "code": f.code, "message": f.message}
            for f in findings
        ],
    }

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote: {output_path}")

    if args.mode == "backward":
        return 0 if backward_ok else 1
    if args.mode == "forward":
        return 0 if forward_ok else 1
    return 0 if (backward_ok and forward_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
