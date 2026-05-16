from typing import List, Dict, Any

def get_benchmark_cases() -> List[Dict[str, Any]]:
    return [
        {
            "case_id": "threshold_approval_001",
            "family": "numeric_threshold",
            "question": "Does this request require manager approval?",
            "expected_answer_contains": ["yes", "$500", "manager approval"],
            "expected_source_ids": ["S1"],
            "must_not_contain": ["no approval required"],
            "packet": {
                "facts": [
                    {
                        "fact_id": "F1",
                        "text": "Expenses over $500 require manager approval.",
                        "protected_spans": ["$500", "require manager approval"],
                        "source_id": "S1"
                    }
                ],
                "governance": {
                    "risk": "medium",
                    "approval_required": True,
                    "flags": ["threshold_rule"]
                },
                "evidence": [
                    {
                        "evidence_id": "E1",
                        "source_id": "S1",
                        "source": "policy://travel-expense-policy#section-4.2",
                        "text": "All expenses over $500 require manager approval before reimbursement."
                    }
                ],
                "gaps": [],
                "contradictions": []
            }
        },
        {
            "case_id": "date_deadline_002",
            "family": "date_deadline",
            "question": "By what date must the report be submitted?",
            "expected_answer_contains": ["October 15", "submitted"],
            "expected_source_ids": ["S2"],
            "must_not_contain": ["no deadline"],
            "packet": {
                "facts": [
                    {
                        "fact_id": "F2",
                        "text": "The annual report must be submitted by October 15.",
                        "protected_spans": ["October 15"],
                        "source_id": "S2"
                    }
                ],
                "governance": {
                    "risk": "low",
                    "approval_required": False,
                    "flags": ["deadline_rule"]
                },
                "evidence": [
                    {
                        "evidence_id": "E2",
                        "source_id": "S2",
                        "source": "policy://reporting-guidelines#section-2",
                        "text": "The annual report is due annually on October 15 without exception."
                    }
                ],
                "gaps": [],
                "contradictions": []
            }
        },
        {
            "case_id": "negation_003",
            "family": "negation",
            "question": "Can contractors access the production database?",
            "expected_answer_contains": ["no", "shall not", "access"],
            "expected_source_ids": ["S3"],
            "must_not_contain": ["can access", "allowed"],
            "packet": {
                "facts": [
                    {
                        "fact_id": "F3",
                        "text": "Contractors shall not access the production database.",
                        "protected_spans": ["shall not access"],
                        "source_id": "S3"
                    }
                ],
                "governance": {
                    "risk": "high",
                    "approval_required": True,
                    "flags": ["access_control"]
                },
                "evidence": [
                    {
                        "evidence_id": "E3",
                        "source_id": "S3",
                        "source": "policy://security-policy#access",
                        "text": "Under no circumstances are external contractors permitted. They shall not access the production database."
                    }
                ],
                "gaps": [],
                "contradictions": []
            }
        },
        {
            "case_id": "exception_004",
            "family": "exception_clause",
            "question": "Are external USB drives allowed?",
            "expected_answer_contains": ["only if", "encrypted", "IT-approved"],
            "expected_source_ids": ["S4"],
            "must_not_contain": ["always allowed", "never allowed"],
            "packet": {
                "facts": [
                    {
                        "fact_id": "F4",
                        "text": "USB drives are prohibited unless they are encrypted and IT-approved.",
                        "protected_spans": ["unless", "encrypted", "IT-approved"],
                        "source_id": "S4"
                    }
                ],
                "governance": {
                    "risk": "high",
                    "approval_required": False,
                    "flags": ["hardware_policy"]
                },
                "evidence": [
                    {
                        "evidence_id": "E4",
                        "source_id": "S4",
                        "source": "policy://hardware#usb",
                        "text": "External storage devices like USB drives are prohibited unless they are strictly encrypted and IT-approved."
                    }
                ],
                "gaps": [],
                "contradictions": []
            }
        },
        {
            "case_id": "contradiction_005",
            "family": "contradiction",
            "question": "What is the reimbursement limit for meals?",
            "expected_answer_contains": ["contradiction", "$50", "$75", "conflict"],
            "expected_source_ids": ["S5", "S6"],
            "must_not_contain": ["is exactly"],
            "packet": {
                "facts": [
                    {
                        "fact_id": "F5",
                        "text": "Meals are reimbursed up to $50.",
                        "protected_spans": ["$50"],
                        "source_id": "S5"
                    },
                    {
                        "fact_id": "F6",
                        "text": "Dinner meals are reimbursed up to $75.",
                        "protected_spans": ["$75"],
                        "source_id": "S6"
                    }
                ],
                "governance": {
                    "risk": "medium",
                    "approval_required": False,
                    "flags": []
                },
                "evidence": [
                    {
                        "evidence_id": "E5",
                        "source_id": "S5",
                        "source": "policy://travel#meals-general",
                        "text": "Standard meal limit is $50 per diem."
                    },
                    {
                        "evidence_id": "E6",
                        "source_id": "S6",
                        "source": "policy://travel#meals-dinner",
                        "text": "For late-night workers, dinner meals are reimbursed up to $75."
                    }
                ],
                "gaps": [],
                "contradictions": ["Conflict between general meal limit ($50) and dinner limit ($75)."]
            }
        },
        {
            "case_id": "gap_006",
            "family": "evidence_gap",
            "question": "What is the policy for international travel?",
            "expected_answer_contains": ["missing", "not provided", "gap", "no evidence"],
            "expected_source_ids": [],
            "must_not_contain": ["the policy is"],
            "packet": {
                "facts": [],
                "governance": {
                    "risk": "low",
                    "approval_required": False,
                    "flags": []
                },
                "evidence": [],
                "gaps": ["No evidence retrieved for 'international travel policy'."],
                "contradictions": []
            }
        }
    ]
