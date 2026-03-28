"""
MNEMOS Installer — Interactive Questions
==========================================

5-question Q/A for profile selection.
Supports interactive (stdin) and non-interactive (dict) modes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class UserAnswers:
    """Structured user answers from the installer Q/A."""
    use_case: str = "other"
    priority: str = "semantic_speed"
    scale: str = "under_100k"
    strict_filters: bool = False
    prefer_manual: bool = False


QUESTIONS = [
    {
        "key": "use_case",
        "prompt": "Primary use case?",
        "options": ["agent_memory", "rag_knowledge_base", "compliance_governed", "other"],
        "default": "other",
    },
    {
        "key": "priority",
        "prompt": "Main priority?",
        "options": ["semantic_speed", "metadata_governance", "simplest_deployment"],
        "default": "semantic_speed",
    },
    {
        "key": "scale",
        "prompt": "Expected scale?",
        "options": ["under_100k", "100k_to_1m", "over_1m"],
        "default": "under_100k",
    },
    {
        "key": "strict_filters",
        "prompt": "Need strict metadata / provenance / tenant filtering?",
        "options": ["yes", "no"],
        "default": "no",
    },
    {
        "key": "prefer_manual",
        "prompt": "Prefer guided install or manual control?",
        "options": ["guided", "manual"],
        "default": "guided",
    },
]


def ask_interactive() -> UserAnswers:
    """Run the interactive Q/A session."""
    answers = UserAnswers()

    print("\n" + "=" * 60)
    print("  MNEMOS Installer — Deployment Profile Selection")
    print("=" * 60 + "\n")

    for q in QUESTIONS:
        options_str = " / ".join(q["options"])
        print(f"  {q['prompt']}")
        print(f"    Options: {options_str}")
        print(f"    Default: {q['default']}")

        while True:
            raw = input(f"    > ").strip().lower()
            if not raw:
                raw = q["default"]

            if q["key"] == "strict_filters":
                if raw in ("yes", "y", "true", "1"):
                    answers.strict_filters = True
                    break
                elif raw in ("no", "n", "false", "0"):
                    answers.strict_filters = False
                    break
            elif q["key"] == "prefer_manual":
                if raw in ("manual", "m"):
                    answers.prefer_manual = True
                    break
                elif raw in ("guided", "g"):
                    answers.prefer_manual = False
                    break
            elif raw in q["options"]:
                setattr(answers, q["key"], raw)
                break

            print(f"    Invalid. Choose from: {options_str}")

        print()

    return answers


def from_dict(data: dict) -> UserAnswers:
    """Build UserAnswers from a dictionary (non-interactive mode)."""
    return UserAnswers(
        use_case=data.get("use_case", "other"),
        priority=data.get("priority", "semantic_speed"),
        scale=data.get("scale", "under_100k"),
        strict_filters=str(data.get("strict_filters", "no")).lower() in ("yes", "true", "1"),
        prefer_manual=str(data.get("prefer_manual", "guided")).lower() in ("manual", "m"),
    )
