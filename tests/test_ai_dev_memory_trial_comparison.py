from pathlib import Path

from tools.compare_ai_dev_memory_trials import compare_trials


MNEMOS = Path("D:/AppWMenmos/trial_results/mnemos_enabled")
NO_MEMORY = Path("D:/AppWoutMenmos/trial_results/no_memory")


def test_external_ai_dev_trials_compare_when_available():
    if not MNEMOS.exists() or not NO_MEMORY.exists():
        return
    summary = compare_trials(MNEMOS, NO_MEMORY)
    assert summary["schema_version"] == "ai-dev-memory-quality-lane-result-v1"
    assert summary["pairwise_summary"]["task_outcome"]["both_completed"] is True
    assert summary["pairwise_summary"]["memory_quality"]["mnemos_used_required_memory_tools"] is True
    assert summary["conditions"]["mnemos_enabled"]["workflow_efficiency"]["memory_tool_calls"] > 0
    assert summary["conditions"]["no_memory"]["workflow_efficiency"]["memory_tool_calls"] == 0
