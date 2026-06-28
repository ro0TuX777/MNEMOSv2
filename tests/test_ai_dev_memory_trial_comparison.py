from pathlib import Path

from tools.compare_ai_dev_memory_trials import compare_trials


MNEMOS = Path("D:/AppWMenmos/trial_results/mnemos_enabled")
NO_MEMORY = Path("D:/AppWoutMenmos/trial_results/no_memory")


def test_external_ai_dev_trials_compare_when_available():
    if not MNEMOS.exists() or not NO_MEMORY.exists():
        return
    summary = compare_trials(MNEMOS, NO_MEMORY)
    assert summary["observed_pattern"]["both_completed"] is True
    assert summary["observed_pattern"]["mnemos_used_required_memory_tools"] is True
    assert summary["metrics"]["mnemos_enabled"]["memory_calls"] > 0
    assert summary["metrics"]["no_memory"]["memory_calls"] == 0
