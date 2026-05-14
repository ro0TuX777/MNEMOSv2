import os
import sys
import json
import random
import uuid
import glob

os.environ['MNEMOS_ECHOFRAME_SHADOW_ENABLED'] = 'true'
os.environ['MNEMOS_ECHOFRAME_SHADOW_MODE'] = 'compact_semantic_minEvidence_hysteresis_v0'
os.environ['MNEMOS_ECHOFRAME_SHADOW_SAMPLE_RATE'] = '1.0'
os.environ['MNEMOS_ECHOFRAME_SHADOW_OUTPUT_DIR'] = 'runtime/echoframe_shadow/'
os.environ['MNEMOS_ECHOFRAME_SHADOW_FAIL_CLOSED'] = 'false'
os.environ['EMBEDDING_MODEL'] = 'all-MiniLM-L6-v2'
os.environ['MNEMOS_EMBEDDING_MODEL'] = 'all-MiniLM-L6-v2'

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from service.app import MnemosRuntime

SCENARIOS = [
    ["How do I start SAM?", "What is the start command?", "Is there a python script?", "Where is the start script located?"],
    ["What is ForgeRoot?", "Explain ForgeRoot plugins.", "How does governance work?", "Tell me about CONCORD.", "What are the 11 stages of CONCORD?"],
    ["How do I deploy MNEMOS v2?", "What are the environment variables?", "What is the Qdrant endpoint?", "Can I use LIGAND with it?"],
    ["Tell me about LIGAND neuro-stack.", "How does VAGUS resource monitor work?", "What is PFC simulation?"],
    ["How do I configure memory RAG?", "Explain the VectorRetrieval process.", "What is the exact score threshold?"],
    ["What is the timeline for MNEMOS v3?", "Is there a roadmap?", "Where is the operator playbook?", "How do I upgrade to beta?"],
    ["Explain the Valo Interoperability platform.", "What is the agent surface?", "How does an agent request resources?"],
    ["What is the ForensicLedger?", "How is HMAC used?", "What is the audit trail database?", "Can I delete it?"],
    ["What are the environment variables for ForgeLedger?", "What is the API key size?", "How do I reset the API key?"],
    ["Tell me about RRF fusion in Qdrant.", "What is QdrantHybridFusion?", "Where is the code for hybrid fusion?"],
    ["What are the SLO reliability benchmarks?", "Where are the reports stored?", "What happens if a benchmark fails?"],
    ["How do I test the application locally?", "Are there e2e tests?", "Where are the tests located?"],
    ["What is the operator playbook?", "What are the first steps for AI devs?", "How do I read context handoff?"],
    ["Who is the lead architect?", "When was v2 launched?", "How do I contribute to the codebase?"]
]

def run_soak():
    print("Initializing MNEMOS Runtime for Phase 3F-R Soak Test...")
    rt = MnemosRuntime()
    rt.initialize()
    
    docs = []
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    doc_paths = glob.glob(os.path.join(base_dir, "docs", "*.md")) + glob.glob(os.path.join(base_dir, "benchmarks", "TEMP", "*.md"))
    
    print(f"Found {len(doc_paths)} markdown files for indexing.")
    
    for path in doc_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > 100:
                    docs.append({
                        "content": content,
                        "source": os.path.basename(path),
                        "metadata": {"doc_type": "markdown", "length": len(content)}
                    })
        except Exception as e:
            pass
            
    print(f"Indexing {len(docs)} real documents...")
    rt.index_documents(docs, {"chunk_size": 2000})
    print("Indexing complete.")
    
    total_calls = 0
    sessions = 20
    for s_idx in range(sessions):
        session_id = f"SOAK_R_{uuid.uuid4().hex[:8]}"
        base_scenario = SCENARIOS[s_idx % len(SCENARIOS)]
        
        # Ensure we have at least 10 turns
        turns = list(base_scenario) * 2 + ["Summarize.", "Explain more.", "Why?", "How?", "Where?"]
        
        for t_idx, query in enumerate(turns[:10]):
            try:
                filters = {"session_id": session_id, "turn_id": f"T{t_idx+1}"}
                rt.search_documents(query, top_k=5, tiers=None, filters=filters, retrieval_mode="semantic", fusion_policy="raw", explain=True)
                total_calls += 1
            except Exception as e:
                print(f"Error on query '{query}': {e}")
                
    print(f"Soak completed. Generated {total_calls} shadow events.")

if __name__ == "__main__":
    run_soak()
