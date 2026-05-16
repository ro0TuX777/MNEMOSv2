import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '../benchmarks/echoframe_phase12a'))
from placement_cases import get_benchmark_cases
from placement_metrics import score_answer

def test_score_correct_answer():
    cases = get_benchmark_cases()
    case = cases[0]
    ans = "Yes, because it is over $500, manager approval is needed. From S1. Warning: high risk."
    scores = score_answer(ans, case)
    assert scores['answer_correct'] == 1.0
    assert scores['source_attribution_correct'] == 1.0
    assert scores['numeric_span_preserved'] == 1.0
    assert scores['placement_quality_score'] > 0.8

def test_score_missing_source():
    cases = get_benchmark_cases()
    case = cases[0]
    ans = "Yes, because it is over $500, manager approval is needed. Warning: high risk."
    scores = score_answer(ans, case)
    assert scores['source_attribution_correct'] == 0.0
