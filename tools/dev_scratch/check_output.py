import json

outputs = json.load(open('eval_results/pit_11a_shadow_outputs.json'))
for o in outputs:
    print(f"QID: {o['question_id']}")
    candidates = o.get("shadow_evaluation", {}).get("candidate_telemetry", [])
    for c in candidates:
        print(f"  {c['selection_decision']} | Align: {c['derived_fact_answer_alignment_score']} | Support: {c['derived_fact_source_support_score']}")
