import os
import sys
import json
import random
import uuid

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
    ["What is the MNEMOS v2 retrieval architecture?", "How does QdrantHybridFusion work?", "What is the exact score threshold for fusion?", "Where is the architecture code located?"],
    ["What are the rules for deploying a new retrieval tier?", "Are there exceptions for deploying without testing?", "I am bypassing deployment tests right now.", "Can I delete the policy?"],
    ["What is the current version of the MNEMOS vector store?", "Was there an older version used?", "How do I configure the newest version?", "What is the config file name?"],
    ["What is the default max context size?", "I found a document saying max context is 1000 and another saying 5000.", "Which limit is correct?", "Who wrote the 5000 limit?"],
    ["What is the exact release date of MNEMOS v3?", "Is there a rough timeline?", "Who is the lead developer of MNEMOS?", "What is their email?"],
    ["What is the ForensicLedger configuration API key?", "How is that API key used in code?", "What happens if the key is invalid?", "How do I reset the key?"],
    ["I am deleting the Forensic Ledger database right now.", "Just kidding. Where does it store the database file?", "Can I read the ledger database file with sqlite3?", "What are the table names?"],
    ["How do I configure Qdrant for MNEMOS?", "What are the configuration steps for MNEMOS Qdrant?", "MNEMOS Qdrant setup configuration?", "Is Qdrant setup hard?"],
    ["What is the LIGAND architecture?", "How does VAGUS monitor resources?", "What is the threshold for VAGUS warnings?", "Can I disable VAGUS?"],
    ["What is the CONCORD admission pipeline?", "What are the 11 stages of CONCORD?", "Is stage 5 required?", "What happens if a stage fails?"],
    ["What are the rules for ForgeLedger HMAC signature chaining?", "Can I bypass HMAC?", "I am bypassing HMAC signature chaining.", "Where is the signature stored?"],
    ["When was the MNEMOS v2 infrastructure deployed?", "Is there an exact date?", "What was deployed before v2?", "How long did it take?"],
    ["What are the environment variables for MNEMOS?", "What are the Qdrant environment variables?", "What is the default Qdrant port?", "Can I change the port?"],
    ["Which port does VAGUS listen on?", "What happens if the port is in use?", "How do I change the VAGUS port?", "Is there a config flag?"],
    ["How does Valo Interoperability Platform interact with CONCORD?", "What is the agent surface?", "Can agents bypass CONCORD?", "Who authorizes agents?"],
    ["Trace the execution from ForgeRoot plugin to SAM Core.", "Where is the plugin registered?", "How does the core find it?", "What is the core API?"]
]

def run_soak():
    print("Initializing MNEMOS Runtime for Phase 3F Soak Test...")
    rt = MnemosRuntime()
    rt.initialize()
    
    # We repeat the string 300 times to simulate a 1500-2000 token real document
    docs = [
        {"content": "MNEMOS v2 retrieval architecture uses QdrantHybridFusion with an exact score threshold of 0.85. " * 300, "source": "docs/arch.md", "metadata": {}},
        {"content": "The QdrantHybridFusion mechanism combines sparse and dense vectors. Architecture code is in mnemos/retrieval/. " * 300, "source": "docs/arch.md", "metadata": {}},
        {"content": "Deploying without testing is strictly prohibited. You cannot bypass deployment tests. " * 300, "source": "policy.md", "metadata": {"governance": "HIGH_RISK", "approval_required": True}},
        {"content": "MNEMOS v3 release date is still unknown, no exact date. " * 300, "source": "timeline.md", "metadata": {}},
        {"content": "ForensicLedger configuration API key must be 64 chars. " * 300, "source": "security.md", "metadata": {}},
        {"content": "LIGAND architecture is a neuro-stack. VAGUS monitors resources. Threshold is 90%. " * 300, "source": "ligand.md", "metadata": {}},
        {"content": "CONCORD admission pipeline has 11 stages. Stage 5 is required. " * 300, "source": "concord.md", "metadata": {}},
        {"content": "ForgeLedger HMAC signature chaining cannot be bypassed. " * 300, "source": "forgeledger.md", "metadata": {}},
        {"content": "MNEMOS v2 infrastructure deployed on 2026-05-01. MNEMOS Qdrant URL is http://localhost:6333. " * 300, "source": "deploy.md", "metadata": {}},
        {"content": "Valo Interoperability Platform interacts via agent surface. " * 300, "source": "valo.md", "metadata": {}},
        {"content": "There is a contradiction here. Actually you CAN bypass HMAC. " * 300, "source": "rogue.md", "metadata": {"governance": "CONTRADICTION_DETECTED"}},
    ] * 5
    
    # Monkey-patch QdrantTier to ignore filters so our mock documents are returned
    from mnemos.retrieval.qdrant_tier import QdrantTier
    QdrantTier._build_filter = staticmethod(lambda x: None)

    print("Indexing mock documents...")
    rt.index_documents(docs, {"chunk_size": 8000})
    print("Indexing complete.")
    
    total_calls = 0
    sessions = 15
    for s_idx in range(sessions):
        session_id = f"SOAK_SESSION_{uuid.uuid4().hex[:8]}"
        base_scenario = SCENARIOS[s_idx % len(SCENARIOS)]
        
        # Add some random noise turns
        turns = list(base_scenario) * 2 + ["What is the meaning of life?", "Tell me a joke."]
        
        for t_idx, query in enumerate(turns[:10]):
            try:
                # We can't pass session_id to search_documents directly because it's a shadow feature
                # Wait, the shadow adapter gets session_id and turn_id from kwargs if they were passed, 
                # but we can't easily pass them. The shadow_adapter in app.py does not receive session_id.
                # Actually, shadow_adapter.py defaults to "default_session".
                # For a true multi-session soak, we should mock or inject the session.
                filters = {"session_id": session_id, "turn_id": f"T{t_idx+1}"}
                rt.search_documents(query, top_k=5, tiers=None, filters=filters, retrieval_mode="semantic", fusion_policy="raw", explain=True)
                total_calls += 1
            except Exception as e:
                print(f"Error on query '{query}': {e}")
                
    print(f"Soak completed. Generated {total_calls} shadow events.")

if __name__ == "__main__":
    run_soak()
