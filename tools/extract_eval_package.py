import json
import csv
import lancedb

base = json.load(open('eval_results/pit_11a_baseline_outputs.json'))
shadow = json.load(open('eval_results/pit_11a_shadow_outputs.json'))

db = lancedb.connect("data/pit11a/lance")
table = db.open_table("mnemos_engrams")

with open('eval_results/pit_11a_full_output_package.md', 'w', encoding='utf-8') as f:
    allowlist = {
        "arxiv:2604.20622v1",
        "arxiv:2604.20601v1",
        "arxiv:NSA-SIGINT-style-manual_2010",
        "arxiv:SignalsVOL_1",
        "arxiv:The Intelligence Oversight Guide (February 2023)"
    }

    f.write('# PIT-11B 12-Question Full Output Package\n\n')
    for b, s in zip(base, shadow):
        q_id = b['question_id']
        q_text = b['query']
        b_ans = b['results'][0]['engram']['content'][:400].replace('\n', ' ') + '...' if b['results'] else 'None'
        b_src = b['results'][0]['engram']['source'] if b['results'] else 'None'
        b_id = b['results'][0]['engram']['id'] if b['results'] else 'None'
        
        # Pre-flight check on baseline results
        if b['results'] and b_src not in allowlist:
            raise AssertionError(f"FAIL: Baseline retrieved from non-allowlisted source: {b_src}")

        s_eval = s.get('shadow_evaluation', {})
        s_packet = s_eval.get('shadow_packet', {})
        s_payload = s_packet.get('derived_evaluation_payload', [])
        s_ans = s_payload[0]['content'] if s_payload else 'None'
        s_nodes = len(s_payload)
        s_src_cit = s_payload[0]['traceability']['source_engram_ids'] if s_payload else []
        s_src_doc = s_payload[0]['traceability'].get('source_uri', 'None') if s_payload else 'None'
        s_span = s_payload[0]['traceability'].get('provenance_span', 'None') if s_payload else 'None'
        s_auth = s_payload[0]['authority_matrix'].get('authority_type', 'None') if s_payload else 'None'
        s_status = s_payload[0]['authority_matrix'].get('confidence_level', 'None') if s_payload else 'None'
        
        if s_payload and s_src_doc not in allowlist and s_src_doc != "mock://synthesis" and s_src_doc != "pit_11a_mock":
            raise AssertionError(f"FAIL: Shadow lane traced to non-allowlisted source: {s_src_doc}")

        # Fetch Source Text Preview
        source_preview = "None"
        if s_src_cit:
            try:
                # LanceDB query
                res = table.search().where(f"id = '{s_src_cit[0]}'").limit(1).to_list()
                if res:
                    source_preview = res[0]["content"][:300].replace("\n", " ") + "..."
            except Exception as e:
                source_preview = f"ERROR fetching: {e}"

        telemetry = f"Baseline latency: {b['latency_ms']:.2f}ms | Shadow latency: {s['latency_ms']:.2f}ms"
        
        candidate_telem = s_eval.get('candidate_telemetry', [])
        
        f.write(f'### QID: {q_id}\n')
        f.write(f'**Question:** {q_text}\n\n')
        
        f.write(f'**[BASELINE]**\n')
        f.write(f'- Answer: {b_ans}\n')
        f.write(f'- source_document: {b_src}\n')
        f.write(f'- source_engram_id: {b_id}\n')
        f.write(f'- corpus_id: PIT_11A_SMALL_CORPUS\n\n')
        
        f.write(f'**[SHADOW]**\n')
        f.write(f'- Answer: {s_ans}\n')
        f.write(f'- Derived_FactNodes: {s_nodes}\n')
        f.write(f'- source_document: {s_src_doc}\n')
        f.write(f'- source_engram_id: {s_src_cit}\n')
        f.write(f'- page/span: {s_span}\n')
        f.write(f'- source_text_preview: {source_preview}\n')
        f.write(f'- derived_fact_status: {s_status}\n')
        f.write(f'- authority_type: {s_auth}\n')
        f.write(f'- corpus_id: PIT_11A_SMALL_CORPUS\n\n')
        
        f.write(f'**[TELEMETRY]**\n')
        f.write(f'{telemetry}\n\n')
        
        f.write(f'**[CANDIDATE SELECTION TELEMETRY]**\n')
        if not candidate_telem:
             f.write("No candidate telemetry available.\n")
        for ct in candidate_telem:
             f.write(f"- Fact ID: {ct.get('fact_id')}\n")
             f.write(f"  - Selection Decision: **{ct.get('selection_decision')}**\n")
             f.write(f"  - Selection Path: **{ct.get('selection_path', 'UNKNOWN')}**\n")
             if ct.get("rescue_reason"):
                 f.write(f"  - Rescue Reason: {ct.get('rescue_reason')}\n")
             f.write(f"  - Rendered Support Decision: **{ct.get('rendered_support_decision', 'N/A')}**\n")
             f.write(f"  - Drop Reason: {ct.get('drop_reason')}\n")
             f.write(f"  - Operator Value Score: {ct.get('operator_value_score', 'N/A')}\n")
             f.write(f"  - Alignment Score: {ct.get('derived_fact_answer_alignment_score')}\n")
             f.write(f"  - Support Score: {ct.get('derived_fact_source_support_score')}\n")
             f.write(f"  - Rendering Score: {ct.get('support_evidence_rendering_quality_score', 'N/A')}\n")
             f.write(f"  - Final Score: {ct.get('final_derived_fact_selection_score')}\n")
             f.write(f"  - Raw Source Preview: {ct.get('raw_source_preview', 'None')}\n")
             f.write(f"  - Evidence Excerpt: {ct.get('support_evidence_excerpt', ct.get('support_evidence_preview', 'None'))}\n")
        f.write('---\n\n')

# Also update the CSV
csv_path = "eval_results/pit_11a_operator_scoring_sheet.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "question_id", "question_type", 
        "baseline_correctness_1_5", "shadow_correctness_1_5", 
        "baseline_evidence_1_5", "shadow_evidence_1_5", 
        "baseline_citation_1_5", "shadow_citation_1_5", 
        "authority_clarity_1_5", "derived_fact_usefulness_0_4", 
        "baseline_missed_fact_recovered_yes_no", 
        "review_burden_delta_-2_to_2", 
        "operator_confidence_delta_-2_to_2", 
        "safety_issue_yes_no", 
        "claim_strength_issue_yes_no", 
        "notes", "decision"
    ])
    for b in base:
        writer.writerow([b["question_id"], "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])

print("Successfully generated package and updated CSV.")
