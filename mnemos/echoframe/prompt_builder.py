"""
EchoFrame Production Prompt Builder
===================================
Standard prompt generation. Strictly forbids derived fact rendering.
"""

from typing import Dict, Any

class SevStop(Exception):
    pass

class PromptBuilder:
    def __init__(self):
        pass

    def build_prompt(self, retrieval_payload: Dict[str, Any], evaluation_mode: bool = False) -> str:
        """
        Builds the production prompt context.
        """
        # PIT-7: Negative guard against derived fact leakage
        derived_results = retrieval_payload.get('derived_results', [])
        if not evaluation_mode and len(derived_results) > 0:
            raise SevStop("DERIVED_FACTS_IN_PRODUCTION_PROMPT")

        primary = retrieval_payload.get('primary_results', [])
        context_lines = [p.get("content", "") for p in primary]
        
        return " ".join(context_lines)
