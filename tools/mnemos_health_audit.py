#!/usr/bin/env python3
"""MNEMOS service health and contract audit.

Reads the registry, validates /health, calls the contract endpoint,
and checks required fields, types, status values, and contract version.

Usage:
  python tools/mnemos_health_audit.py
  python tools/mnemos_health_audit.py --registry registry/services.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable


def _find_root(start: Path) -> Path:
    """Walk up from *start* until we find registry/services.json."""
    for candidate in [start, *start.parents]:
        if (candidate / "registry" / "services.json").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate project root containing registry/services.json"
    )


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_type(value: Any, expected: str) -> bool:
    if expected == "str":
        return isinstance(value, str)
    if expected == "list":
        return isinstance(value, list)
    if expected == "nullable_str":
        return value is None or isinstance(value, str)
    return False


def _iter_base_url_envs(service: Dict[str, Any]) -> Iterable[str]:
    envs = service.get("base_url_envs")
    if isinstance(envs, list):
        for env in envs:
            if isinstance(env, str) and env.strip():
                yield env.strip()
    # Legacy single-key fallback
    legacy = service.get("base_url_env")
    if isinstance(legacy, str) and legacy.strip():
        yield legacy.strip()


def _resolve_base_url(service: Dict[str, Any]) -> str:
    default = str(service["default_base_url"]).rstrip("/")
    for env_name in _iter_base_url_envs(service):
        val = os.getenv(env_name)
        if val:
            return val.rstrip("/")
    return default


def main(argv: list[str] | None = None) -> int:
    # Lazy import — only needed at runtime
    try:
        import requests  # noqa: F811
    except ImportError:
        print("[ERROR] 'requests' package is required.  pip install requests")
        return 1

    parser = argparse.ArgumentParser(description="MNEMOS health & contract audit")
    parser.add_argument(
        "--registry", type=str, default=None,
        help="Path to registry/services.json (auto-detected if omitted)",
    )
    args = parser.parse_args(argv)

    if args.registry:
        registry_path = Path(args.registry)
        root = registry_path.parent.parent
    else:
        root = _find_root(Path(__file__).resolve().parent)
        registry_path = root / "registry" / "services.json"

    registry = _load_json(registry_path)
    services = registry.get("services", [])
    if not services:
        print("[WARNING] No services found in registry.")
        return 0

    failures = 0

    for svc in services:
        name = svc["name"]
        base_url = _resolve_base_url(svc)
        health_url = base_url + svc["health_path"]
        contract_url = base_url + svc["contract_path"]
        contract_file = root / svc["contract_file"]
        contract = _load_json(contract_file)

        print(f"\n== {name} ==")

        # 1) Health check
        try:
            health_resp = requests.get(health_url, timeout=3.0)
            health_resp.raise_for_status()
            print(f"  [OK]   health: {health_url}")
        except Exception as exc:
            print(f"  [FAIL] health: {health_url} -> {exc}")
            failures += 1
            continue

        # 2) Contract endpoint
        try:
            payload_resp = requests.get(contract_url, timeout=5.0)
            payload_resp.raise_for_status()
            payload = payload_resp.json()
            print(f"  [OK]   contract endpoint: {contract_url}")
        except Exception as exc:
            print(f"  [FAIL] contract endpoint: {contract_url} -> {exc}")
            failures += 1
            continue

        # 3) Field validation
        required_fields = contract.get("required_fields", {})
        field_errors = []
        for field_name, expected_type in required_fields.items():
            if field_name not in payload:
                field_errors.append(f"missing field: {field_name}")
                continue
            if not _validate_type(payload[field_name], expected_type):
                field_errors.append(
                    f"type mismatch: {field_name} expected {expected_type}, "
                    f"got {type(payload[field_name]).__name__}"
                )

        # 4) Status value check
        allowed_status = set(contract.get("allowed_status", []))
        if payload.get("status") not in allowed_status:
            field_errors.append(f"invalid status: {payload.get('status')}")

        # 5) Contract version drift
        if payload.get("contract_version") != contract.get("contract_version"):
            field_errors.append(
                f"contract drift: expected {contract.get('contract_version')}, "
                f"got {payload.get('contract_version')}"
            )

        if field_errors:
            failures += 1
            print("  [FAIL] contract validation")
            for err in field_errors:
                print(f"         - {err}")
        else:
            print("  [OK]   contract validation")

    print()
    if failures:
        print(f"Audit completed with {failures} failure(s).")
        return 1
    print("Audit completed successfully. ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
