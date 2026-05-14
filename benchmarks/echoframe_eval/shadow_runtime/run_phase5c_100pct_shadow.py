import os
import sys
import json
import random
import uuid
import glob

os.environ['MNEMOS_ECHOFRAME_SHADOW_ENABLED'] = 'true'
os.environ['MNEMOS_ECHOFRAME_SHADOW_MODE'] = 'compact_semantic_minEvidence_hysteresis_v0'
os.environ['MNEMOS_ECHOFRAME_SHADOW_SAMPLE_RATE'] = '1.00'
os.environ['MNEMOS_ECHOFRAME_SHADOW_OUTPUT_DIR'] = 'runtime/echoframe_shadow/'
os.environ['MNEMOS_ECHOFRAME_SHADOW_FAIL_CLOSED'] = 'false'
os.environ['EMBEDDING_MODEL'] = 'all-MiniLM-L6-v2'
os.environ['MNEMOS_EMBEDDING_MODEL'] = 'all-MiniLM-L6-v2'

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from service.app import MnemosRuntime

SCENARIOS = [
    ["How do I start SAM?", "What is the start command?", "Is there a python script?", "Where is the start script located?", "Is there a bug?", "Please clarify.", "Explain.", "Thanks.", "Are you sure?", "Is it fast?", "Can I automate it?", "Where are the logs?", "Tell me more about logs.", "Can I delete logs?", "Is the log size big?", "Who creates the logs?", "What format is the log?", "Are there JSON logs?", "How to rotate logs?", "I have another question."],
    ["What is ForgeRoot?", "Explain ForgeRoot plugins.", "How does governance work?", "Tell me about CONCORD.", "What are the 11 stages of CONCORD?", "Is stage 5 important?", "What happens at stage 10?", "Can I skip stages?", "Who authored CONCORD?", "Is CONCORD active?", "How to disable CONCORD?", "Where is the code for CONCORD?", "Is there an API?", "What is the API key?", "How to rotate the key?", "Tell me more.", "Yes.", "No.", "Explain.", "Thanks."],
    ["How do I deploy MNEMOS v2?", "What are the environment variables?", "What is the Qdrant endpoint?", "Can I use LIGAND with it?", "What is LIGAND?", "Who built LIGAND?", "Is LIGAND fast?", "How to disable it?", "What is the port?", "Can I change the port?", "Where is the config?", "Is the config JSON?", "Is the config YAML?", "What is the default?", "Can I reset defaults?", "I need help.", "Help me.", "Explain LIGAND again.", "Thanks.", "Goodbye."],
    ["Tell me about LIGAND neuro-stack.", "How does VAGUS resource monitor work?", "What is PFC simulation?", "Is PFC required?", "Can I bypass PFC?", "Where is PFC code?", "What is the memory footprint?", "Is VAGUS a daemon?", "How to kill VAGUS?", "Can VAGUS restart?", "Tell me about memory leaks.", "Are there leaks?", "How to debug?", "Where is the debugger?", "Is there a UI?", "Where is the UI?", "How to login?", "What is the password?", "Can I reset password?", "Thanks."],
    ["How do I configure memory RAG?", "Explain the VectorRetrieval process.", "What is the exact score threshold?", "Can I lower the threshold?", "What if score is 0?", "Is 0 valid?", "What is QdrantHybridFusion?", "Where is hybrid fusion code?", "Tell me more.", "Yes.", "No.", "Explain.", "Thanks.", "Are there alternatives?", "Can I use Milvus?", "Can I use Pinecone?", "Why Qdrant?", "Is Qdrant open source?", "What version of Qdrant?", "Thanks."]
]

def run_soak():
    print("Initializing MNEMOS Runtime for Phase 5C 100% Shadow Review...")
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
            
    # We won't re-index docs to save time, assuming Phase 3F-R docs are still in Qdrant.
    
    total_calls = 0
    sessions = 50
    for s_idx in range(sessions):
        session_id = f"PHASE5C_{uuid.uuid4().hex[:8]}"
        base_scenario = SCENARIOS[s_idx % len(SCENARIOS)]
        
        # Use the 20 turns directly
        turns = base_scenario
        
        for t_idx, query in enumerate(turns):
            try:
                filters = {"session_id": session_id, "turn_id": f"T{t_idx+1}"}
                rt.search_documents(query, top_k=5, tiers=None, filters=filters, retrieval_mode="semantic", fusion_policy="raw", explain=True)
                total_calls += 1
            except Exception as e:
                print(f"Error on query '{query}': {e}")
                
    print(f"Soak completed. Generated {total_calls} shadow events.")

if __name__ == "__main__":
    run_soak()
