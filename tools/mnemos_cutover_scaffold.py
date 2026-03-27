#!/usr/bin/env python3
"""MNEMOS Cutover Scaffold — generate a staged rollout manifest for MNEMOS adoption.

Creates a cutover manifest for apps transitioning from another memory backend
(Redis, Elasticsearch, etc.) to MNEMOS, with plan/promote/rollback lifecycle.

Usage:
  python tools/mnemos_cutover_scaffold.py --app my-app
  python tools/mnemos_cutover_scaffold.py --app my-app --output docs/cutover_manifest.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _build_manifest(app_name: str) -> dict:
    return {
        "feature": f"mnemos-memory-{app_name}",
        "app": app_name,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": f"Cutover manifest for {app_name} migrating to MNEMOS memory service",
        "stages": [
            {
                "name": "shadow",
                "description": "Write to both old backend and MNEMOS; read only from old backend",
                "traffic_pct": 0,
                "commands": {
                    "deploy": f"echo 'Enable MNEMOS shadow writes for {app_name}'",
                    "verify": "python tools/mnemos_health_audit.py",
                    "rollback": f"echo 'Disable MNEMOS shadow writes for {app_name}'",
                },
            },
            {
                "name": "canary_5",
                "description": "Route 5% of reads to MNEMOS",
                "traffic_pct": 5,
                "commands": {
                    "deploy": f"echo 'Route 5% of {app_name} reads to MNEMOS'",
                    "verify": "python tools/mnemos_health_audit.py",
                    "rollback": f"echo 'Revert {app_name} reads to old backend'",
                },
            },
            {
                "name": "canary_25",
                "description": "Route 25% of reads to MNEMOS",
                "traffic_pct": 25,
                "commands": {
                    "deploy": f"echo 'Route 25% of {app_name} reads to MNEMOS'",
                    "verify": "python tools/mnemos_health_audit.py",
                    "rollback": f"echo 'Revert {app_name} reads to old backend'",
                },
            },
            {
                "name": "canary_50",
                "description": "Route 50% of reads to MNEMOS",
                "traffic_pct": 50,
                "commands": {
                    "deploy": f"echo 'Route 50% of {app_name} reads to MNEMOS'",
                    "verify": "python tools/mnemos_health_audit.py",
                    "rollback": f"echo 'Revert {app_name} reads to old backend'",
                },
            },
            {
                "name": "full",
                "description": "Route 100% of traffic to MNEMOS; decommission old backend",
                "traffic_pct": 100,
                "commands": {
                    "deploy": f"echo 'Full cutover: {app_name} -> MNEMOS'",
                    "verify": "python tools/mnemos_health_audit.py",
                    "rollback": f"echo 'Emergency rollback: {app_name} -> old backend'",
                },
            },
        ],
        "gates": {
            "pre_promote": [
                "health_audit_pass",
                "contract_validation_pass",
            ],
            "post_promote": [
                "smoke_test_pass",
                "latency_within_budget",
            ],
        },
        "rollback_trigger": "Any gate failure or manual operator decision",
        "notes": [
            "Replace echo commands with real deployment/config commands",
            "Add latency and error-rate thresholds in gates.post_promote",
            "State is tracked in <manifest>.state.json",
        ],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate MNEMOS cutover manifest")
    parser.add_argument("--app", required=True, help="Consumer application name")
    parser.add_argument("--output", default="", help="Output path (default: docs/mnemos_cutover_<app>.json)")
    args = parser.parse_args(argv)

    manifest = _build_manifest(args.app)

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = Path(f"docs/mnemos_cutover_{args.app.replace('-', '_')}.json").resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"✅ Cutover manifest written to: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Replace placeholder commands with real deploy/rollback commands")
    print(f"  2. Start with 'shadow' stage for dual-write validation")
    print(f"  3. Promote through canary stages, validating at each gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
