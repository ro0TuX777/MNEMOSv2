import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../benchmarks/echoframe_phase12b'))
from format_cases import get_format_cases
from format_metrics import score_format_answer

def test_score_format_correct_answer():
    cases = get_format_cases()
    case = cases[0]
    ans = "Yes, because it is over $500, manager approval is needed. From S1. Warning: high risk."
    scores = score_format_answer(ans, case)
    assert scores['answer_correct'] == 1.0
    assert scores['source_attribution_correct'] == 1.0
    assert scores['numeric_span_preserved'] == 1.0
    assert scores['format_quality_score'] > 0.8
    assert scores['parser_reliability'] == 1.0

def test_score_format_missing_source():
    cases = get_format_cases()
    case = cases[0]
    ans = "Yes, because it is over $500, manager approval is needed. Warning: high risk."
    scores = score_format_answer(ans, case)
    assert scores['source_attribution_correct'] == 0.0
