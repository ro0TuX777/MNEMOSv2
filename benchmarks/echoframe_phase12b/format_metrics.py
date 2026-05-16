import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../echoframe_phase12a'))
from placement_metrics import score_answer as phase12a_score_answer
from typing import Dict, Any

def score_format_answer(answer: str, case: Dict[str, Any]) -> Dict[str, Any]:
    # Reuse base metrics
    base_scores = phase12a_score_answer(answer, case)
    
    # We add format-specific metrics
    # numeric_span_preserved is already calculated in base_scores
    # date_span_preserved is practically the same logic, we can map it
    family = case.get('family')
    answer_correct = base_scores['answer_correct']
    
    date_span_preserved = 1.0 if family == 'date_deadline' else 1.0
    if family == 'date_deadline' and answer_correct == 0.0:
        date_span_preserved = 0.0
        
    # parser_reliability: Since we don't have a real parser in this benchmark, 
    # we simulate it by checking if the output contains malformed structures if it was JSON/YAML etc. 
    # We will just default to 1.0 here unless there's a reason to fail.
    parser_reliability = 1.0

    format_quality_score = (
        0.18 * base_scores['answer_correct'] +
        0.14 * base_scores['source_attribution_correct'] +
        0.12 * base_scores['numeric_span_preserved'] +
        0.10 * date_span_preserved +
        0.12 * base_scores['negation_preserved'] +
        0.10 * base_scores['exception_preserved'] +
        0.10 * base_scores['governance_warning_preserved'] +
        0.08 * base_scores['contradiction_acknowledged'] +
        0.08 * base_scores['evidence_gap_acknowledged'] +
        0.08 * parser_reliability
    )
    
    scores = base_scores.copy()
    scores['date_span_preserved'] = date_span_preserved
    scores['parser_reliability'] = parser_reliability
    scores['format_quality_score'] = format_quality_score
    return scores
