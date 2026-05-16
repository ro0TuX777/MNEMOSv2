from typing import Dict, Any

def score_answer(answer: str, case: Dict[str, Any]) -> Dict[str, Any]:
    answer_lower = answer.lower()
    
    answer_correct = 1.0
    for term in case.get('expected_answer_contains', []):
        if term.lower() not in answer_lower:
            answer_correct = 0.0
            break
            
    for term in case.get('must_not_contain', []):
        if term.lower() in answer_lower:
            answer_correct = 0.0
            break

    source_attribution_correct = 1.0
    for sid in case.get('expected_source_ids', []):
        if sid.lower() not in answer_lower:
            source_attribution_correct = 0.0
            break
            
    family = case.get('family')
    
    numeric_span_preserved = 1.0 if family in ['numeric_threshold', 'date_deadline'] else 1.0
    if family in ['numeric_threshold', 'date_deadline'] and answer_correct == 0.0:
        numeric_span_preserved = 0.0

    negation_preserved = 1.0 if family == 'negation' else 1.0
    if family == 'negation' and answer_correct == 0.0:
        negation_preserved = 0.0

    exception_preserved = 1.0 if family == 'exception_clause' else 1.0
    if family == 'exception_clause' and answer_correct == 0.0:
        exception_preserved = 0.0

    contradiction_acknowledged = 1.0 if family == 'contradiction' else 1.0
    if family == 'contradiction' and answer_correct == 0.0:
        contradiction_acknowledged = 0.0

    evidence_gap_acknowledged = 1.0 if family == 'evidence_gap' else 1.0
    if family == 'evidence_gap' and answer_correct == 0.0:
        evidence_gap_acknowledged = 0.0

    gov = case['packet'].get('governance', {})
    governance_warning_preserved = 1.0
    if gov.get('approval_required') or gov.get('risk') == 'high':
        warning_words = ['approval', 'warning', 'high risk', 'manager', 'not allowed']
        if not any(w in answer_lower for w in warning_words):
            governance_warning_preserved = 0.0

    placement_quality_score = (
        0.20 * answer_correct +
        0.15 * source_attribution_correct +
        0.15 * numeric_span_preserved +
        0.15 * negation_preserved +
        0.10 * exception_preserved +
        0.10 * governance_warning_preserved +
        0.075 * contradiction_acknowledged +
        0.075 * evidence_gap_acknowledged
    )

    return {
        "answer_correct": answer_correct,
        "source_attribution_correct": source_attribution_correct,
        "numeric_span_preserved": numeric_span_preserved,
        "negation_preserved": negation_preserved,
        "exception_preserved": exception_preserved,
        "contradiction_acknowledged": contradiction_acknowledged,
        "evidence_gap_acknowledged": evidence_gap_acknowledged,
        "governance_warning_preserved": governance_warning_preserved,
        "placement_quality_score": placement_quality_score
    }
