"""
MNEMOS Installer — Entry Point
=================================

Usage:
    python -m installer                                    # Interactive
    python -m installer --profile core_memory_appliance    # Non-interactive
    python -m installer --profile governance_native        # Non-interactive
    python -m installer --dry-run                          # Preview only
"""

import argparse
import sys
from pathlib import Path

from installer.profiles import PROFILES, list_profiles, get_profile
from installer.questions import ask_interactive, from_dict, UserAnswers
from installer.probes import run_probes, ProbeResults
from installer.recommend import recommend, Recommendation
from installer.render import render_compose, render_env, render_manifest

FUSION_POLICIES = ["semantic_dominant", "balanced", "lexical_dominant"]


def _print_header():
    print()
    print("=" * 44)
    print("  MNEMOS Deployment Installer")
    print("  Contract-Governed Memory Service")
    print("=" * 44)
    print()


def _print_recommendation(rec: Recommendation):
    print("\n" + "=" * 60)
    print(f"  Recommended Profile: {rec.profile.display_name}")
    print(f"  Confidence: {rec.confidence}")
    print("=" * 60)

    print("\n  Why:")
    for r in rec.reasons:
        print(f"    + {r}")

    if rec.warnings:
        print("\n  Warnings:")
        for w in rec.warnings:
            print(f"    {w}")

    if rec.alternatives:
        print(f"\n  Also viable: {', '.join(rec.alternatives)}")

    print(f"\n  Stack: {rec.profile.description}")
    print(f"  Containers: {rec.profile.containers}")
    print(f"  Best for: {rec.profile.best_for}")
    print()


def _print_probes(probes: ProbeResults):
    print("  Host Detection:")
    print(f"    GPU:     {'[OK] ' + probes.gpu_name if probes.gpu_available else '[NO] Not detected'}")
    if probes.vram_mb:
        print(f"    VRAM:    {probes.vram_mb} MB")
    print(f"    RAM:     {probes.ram_gb} GB")
    print(f"    Disk:    {probes.disk_free_gb} GB free")
    print(f"    Docker:  {'[OK]' if probes.docker_available else '[NO]'}")
    print(f"    NVIDIA:  {'[OK]' if probes.nvidia_runtime else '[NO]'}")
    print(f"    CPU:     {probes.cpu_cores} cores")
    print(f"    OS:      {probes.os_name}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="MNEMOS Deployment Profile Installer",
        prog="python -m installer",
    )
    parser.add_argument(
        "--profile", type=str, choices=list(PROFILES.keys()),
        help="Skip Q/A and use this profile directly",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview recommendation without generating files",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory for generated files (default: current)",
    )
    parser.add_argument(
        "--retrieval-mode", type=str, choices=["semantic", "hybrid"], default="semantic",
        help="Retrieval mode written into generated .env.mnemos (default: semantic)",
    )
    parser.add_argument(
        "--fusion-policy", type=str, choices=FUSION_POLICIES, default="balanced",
        help="Fusion policy for hybrid mode in generated .env.mnemos (default: balanced)",
    )
    parser.add_argument(
        "--lexical-top-k", type=int, default=25,
        help="Hybrid lexical candidate top-k in generated .env.mnemos (default: 25)",
    )
    parser.add_argument(
        "--semantic-top-k", type=int, default=25,
        help="Hybrid semantic candidate top-k in generated .env.mnemos (default: 25)",
    )
    parser.add_argument(
        "--explain-default", action="store_true",
        help="Set MNEMOS_EXPLAIN_DEFAULT=true in generated .env.mnemos",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.lexical_top_k < 1:
        print("  [ERROR] --lexical-top-k must be >= 1")
        sys.exit(1)
    if args.semantic_top_k < 1:
        print("  [ERROR] --semantic-top-k must be >= 1")
        sys.exit(1)

    _print_header()

    # Step 1: Get user answers
    if args.profile:
        print(f"  Using profile: {args.profile}\n")
        profile = get_profile(args.profile)
        if not profile:
            print(f"  [ERROR] Unknown profile: {args.profile}")
            sys.exit(1)
        answers = UserAnswers(prefer_manual=(args.profile == "custom_manual"))
    else:
        answers = ask_interactive()

    # Step 2: Run system probes
    print("  Running system probes...")
    probes = run_probes()
    _print_probes(probes)

    # Step 3: Get recommendation
    if args.profile:
        # Direct profile selection — build a simple recommendation
        rec = Recommendation(
            profile=PROFILES[args.profile],
            confidence="high",
            reasons=["Profile selected directly via --profile flag"],
        )
        # Still check probes for warnings
        if not probes.gpu_available:
            rec.warnings.append("[WARN] No GPU detected")
        if not probes.docker_available:
            rec.warnings.append("[WARN] Docker not detected")
    else:
        rec = recommend(answers, probes)

    _print_recommendation(rec)

    if args.dry_run:
        print("  [dry-run] No files generated.\n")
        return

    # Step 4: Confirm
    if not args.profile:
        confirm = input("  Accept this profile? (yes/no) > ").strip().lower()
        if confirm not in ("yes", "y", ""):
            print("\n  Available profiles:")
            for p in list_profiles():
                print(f"    - {p.name}: {p.display_name} — {p.best_for}")
            alt = input("\n  Enter profile name (or 'quit'): ").strip()
            if alt == "quit" or alt not in PROFILES:
                print("  Cancelled.\n")
                return
            rec = Recommendation(
                profile=PROFILES[alt],
                confidence="high",
                reasons=["User manually selected profile"],
            )

    # Step 5: Generate files
    print("  Generating configuration files...")

    compose_path = render_compose(rec.profile, output_dir)
    print(f"    [OK] {compose_path}")

    env_path = render_env(
        rec.profile,
        output_dir,
        retrieval_mode=args.retrieval_mode,
        fusion_policy=args.fusion_policy,
        lexical_top_k=args.lexical_top_k,
        semantic_top_k=args.semantic_top_k,
        explain_default=args.explain_default,
    )
    print(f"    [OK] {env_path}")

    manifest_path = render_manifest(
        rec,
        answers,
        probes,
        output_dir,
        retrieval_mode=args.retrieval_mode,
        fusion_policy=args.fusion_policy,
        lexical_top_k=args.lexical_top_k,
        semantic_top_k=args.semantic_top_k,
        explain_default=args.explain_default,
    )
    print(f"    [OK] {manifest_path}")

    print(f"\n  [OK] Installation complete!")
    print(f"  Profile: {rec.profile.display_name}")
    print(f"  Retrieval mode: {args.retrieval_mode}")
    print(f"  Fusion policy: {args.fusion_policy}")
    print(f"\n  Next steps:")
    print(f"    1. Review: cat {env_path}")
    print(f"    2. Start:  docker compose -f {compose_path} up -d --build")
    print(f"    3. Verify: python tools/mnemos_health_audit.py")
    print()


if __name__ == "__main__":
    main()
