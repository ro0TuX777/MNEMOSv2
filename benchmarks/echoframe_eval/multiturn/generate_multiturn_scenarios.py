import json
import os

SCENARIOS = [
    {
        "scenario_id": "SCEN_01",
        "description": "Broad architecture to narrowing down to thresholds",
        "turns": [
            {
                "turn_id": "T1",
                "query": "What is the MNEMOS v2 retrieval architecture?",
                "expected_behavior": "Return broad architectural docs",
                "risk_level": "low",
                "expected_context_behavior": "expand"
            },
            {
                "turn_id": "T2",
                "query": "How does QdrantHybridFusion work in the architecture?",
                "expected_behavior": "Narrow to Qdrant specific logic",
                "risk_level": "low",
                "expected_context_behavior": "compress"
            },
            {
                "turn_id": "T3",
                "query": "What is the exact score threshold for Qdrant fusion?",
                "expected_behavior": "Retrieve exact threshold facts",
                "risk_level": "low",
                "expected_context_behavior": "compress"
            }
        ]
    },
    {
        "scenario_id": "SCEN_02",
        "description": "Policy query escalating into high-risk approval constraint",
        "turns": [
            {
                "turn_id": "T1",
                "query": "What are the rules for deploying a new retrieval tier?",
                "expected_behavior": "Return deployment policy",
                "risk_level": "medium",
                "expected_context_behavior": "expand"
            },
            {
                "turn_id": "T2",
                "query": "Are there any exceptions for deploying without testing?",
                "expected_behavior": "Return exceptions and negations",
                "risk_level": "medium",
                "expected_context_behavior": "retain"
            },
            {
                "turn_id": "T3",
                "query": "I am bypassing the deployment tests to push to production now.",
                "expected_behavior": "High risk flag, approval required",
                "risk_level": "high",
                "expected_context_behavior": "expand"
            }
        ]
    },
    {
        "scenario_id": "SCEN_03",
        "description": "Stale/current disambiguation across multiple turns",
        "turns": [
            {
                "turn_id": "T1",
                "query": "What is the current version of the MNEMOS vector store?",
                "expected_behavior": "Return current version info",
                "risk_level": "medium",
                "expected_context_behavior": "expand"
            },
            {
                "turn_id": "T2",
                "query": "Was there an older version we used previously like v1.1?",
                "expected_behavior": "Retrieve stale docs for comparison",
                "risk_level": "medium",
                "expected_context_behavior": "expand"
            },
            {
                "turn_id": "T3",
                "query": "How do I configure the newest version?",
                "expected_behavior": "Ensure only current is used, drop stale",
                "risk_level": "medium",
                "expected_context_behavior": "compress"
            }
        ]
    },
    {
        "scenario_id": "SCEN_04",
        "description": "Contradiction discovery and retention",
        "turns": [
            {
                "turn_id": "T1",
                "query": "What is the default max context size for EchoFrame?",
                "expected_behavior": "Return default config",
                "risk_level": "low",
                "expected_context_behavior": "expand"
            },
            {
                "turn_id": "T2",
                "query": "I found a document saying max context is 1000 and another saying 5000.",
                "expected_behavior": "Trigger contradiction handling",
                "risk_level": "high",
                "expected_context_behavior": "expand"
            },
            {
                "turn_id": "T3",
                "query": "Which of those two limits is correct?",
                "expected_behavior": "Maintain contradiction context",
                "risk_level": "high",
                "expected_context_behavior": "retain"
            }
        ]
    },
    {
        "scenario_id": "SCEN_05",
        "description": "Unknown/insufficient evidence persisting",
        "turns": [
            {
                "turn_id": "T1",
                "query": "What is the exact release date of MNEMOS v3?",
                "expected_behavior": "Insufficient evidence gap",
                "risk_level": "medium",
                "expected_context_behavior": "reset"
            },
            {
                "turn_id": "T2",
                "query": "Is there any rough timeline at all for v3?",
                "expected_behavior": "Still insufficient evidence",
                "risk_level": "medium",
                "expected_context_behavior": "retain"
            },
            {
                "turn_id": "T3",
                "query": "Who is the lead developer of MNEMOS?",
                "expected_behavior": "Retrieve info on author, unrelated to v3",
                "risk_level": "low",
                "expected_context_behavior": "reset"
            }
        ]
    },
    {
        "scenario_id": "SCEN_06",
        "description": "API lookup followed by implementation details",
        "turns": [
            {
                "turn_id": "T1",
                "query": "What is the ForensicLedger configuration API key?",
                "expected_behavior": "Return API key info",
                "risk_level": "low",
                "expected_context_behavior": "expand"
            },
            {
                "turn_id": "T2",
                "query": "How is that API key used in code?",
                "expected_behavior": "Provide code snippet for the key",
                "risk_level": "low",
                "expected_context_behavior": "retain"
            },
            {
                "turn_id": "T3",
                "query": "What happens if the key is invalid?",
                "expected_behavior": "Explain failure modes",
                "risk_level": "low",
                "expected_context_behavior": "compress"
            }
        ]
    },
    {
        "scenario_id": "SCEN_07",
        "description": "High-risk escalation then de-escalation",
        "turns": [
            {
                "turn_id": "T1",
                "query": "I am deleting the Forensic Ledger database right now.",
                "expected_behavior": "High risk contradiction/approval",
                "risk_level": "high",
                "expected_context_behavior": "expand"
            },
            {
                "turn_id": "T2",
                "query": "Just kidding. Where does it store the database file?",
                "expected_behavior": "Risk decreases, return path",
                "risk_level": "low",
                "expected_context_behavior": "retain" # retain high risk for a bit just in case
            },
            {
                "turn_id": "T3",
                "query": "Can I read the ledger database file with sqlite3?",
                "expected_behavior": "Return DB read info",
                "risk_level": "low",
                "expected_context_behavior": "compress"
            }
        ]
    },
    {
        "scenario_id": "SCEN_08",
        "description": "Same query phrased differently",
        "turns": [
            {
                "turn_id": "T1",
                "query": "How do I configure Qdrant for MNEMOS?",
                "expected_behavior": "Return config",
                "risk_level": "low",
                "expected_context_behavior": "expand"
            },
            {
                "turn_id": "T2",
                "query": "What are the configuration steps for MNEMOS Qdrant?",
                "expected_behavior": "Same result basically",
                "risk_level": "low",
                "expected_context_behavior": "retain"
            },
            {
                "turn_id": "T3",
                "query": "MNEMOS Qdrant setup configuration?",
                "expected_behavior": "Same result basically",
                "risk_level": "low",
                "expected_context_behavior": "retain"
            }
        ]
    }
]

def main():
    out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "multiturn_scenarios.jsonl")
    
    with open(out_file, "w", encoding="utf-8") as f:
        for scen in SCENARIOS:
            f.write(json.dumps(scen) + "\n")
            
    print(f"Generated {len(SCENARIOS)} multi-turn scenarios in {out_file}")

if __name__ == "__main__":
    main()
