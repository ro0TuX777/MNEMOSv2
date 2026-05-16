import json
import os

queries = []

# 10 semantic recall queries
for i in range(10):
    queries.append({
        "query_id": f"semantic_{i+1:03d}",
        "query": f"Explain the broad strategic goals and methodologies outlined in document {i+1}.",
        "query_type": "semantic_recall",
        "expected_dataset": "ToLearn" if i < 5 else "SIGINT",
        "expected_source_contains": [],
        "expected_terms": [],
        "should_have_answer": True
    })

# 10 acronym/exact-term queries
acronyms = ["SIGINT", "COMINT", "ELINT", "NSA", "CIA", "OSINT", "HUMINT", "MASINT", "GEOINT", "FISINT"]
for i, acr in enumerate(acronyms):
    queries.append({
        "query_id": f"acronym_{i+1:03d}",
        "query": f"What are the reporting channels and definitions for {acr} operations?",
        "query_type": "acronym_exact_term",
        "expected_dataset": "SIGINT",
        "expected_source_contains": [],
        "expected_terms": [acr],
        "should_have_answer": True
    })

# 10 source-specific queries
for i in range(10):
    queries.append({
        "query_id": f"source_{i+1:03d}",
        "query": f"According to the source document {i+1}, what is the main objective?",
        "query_type": "source_specific",
        "expected_dataset": "ToLearn" if i < 5 else "SIGINT",
        "expected_source_contains": [f"doc_{i+1}"],
        "expected_terms": [],
        "should_have_answer": True
    })

# 10 policy/procedure queries
policies = ["retention", "compliance", "authorization", "legal", "audit", "penalty", "access control", "classification", "data sharing", "destruction"]
for i, pol in enumerate(policies):
    queries.append({
        "query_id": f"policy_{i+1:03d}",
        "query": f"What is the official procedure regarding {pol}?",
        "query_type": "policy_procedure",
        "expected_dataset": "SIGINT",
        "expected_source_contains": [],
        "expected_terms": [pol],
        "should_have_answer": True
    })

# 10 hard-negative/evidence-gap queries
gaps = ["lunar mining safety waivers", "recipe for chocolate cake", "1994 World Cup winner", "Jupiter's moons count", "capital of Australia", "nuclear reactor build guide", "common cold symptoms", "President of France", "boiling point of water", "chess rules"]
for i, gap in enumerate(gaps):
    queries.append({
        "query_id": f"gap_{i+1:03d}",
        "query": f"What does this corpus say about {gap}?",
        "query_type": "evidence_gap",
        "expected_dataset": None,
        "expected_source_contains": [],
        "expected_terms": [],
        "should_have_answer": False
    })

out_path = "benchmarks/turbovec/query_sets/tq13_real_pdf_queries.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(queries, f, indent=2)

print(f"Generated {len(queries)} queries to {out_path}")
