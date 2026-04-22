"""
Rerank Policy Engine
=====================

Handles the dynamic conditional reranking logic, including candidate minimums,
family allowlists, circuit breakers, and budget tracking.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List
import collections
import time
import logging

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Monitors rerank errors, timeouts, and latency drops."""
    
    def __init__(self, max_timeout_rate: float, max_error_rate: float, window_size: int = 100):
        self.max_timeout_rate = max_timeout_rate
        self.max_error_rate = max_error_rate
        self.window_size = window_size
        
        # ring buffers
        self._results = collections.deque(maxlen=window_size)
    
    def record_success(self):
        self._results.append("success")
        
    def record_timeout(self):
        self._results.append("timeout")

    def record_error(self):
        self._results.append("error")

    @property
    def is_open(self) -> bool:
        """True if the breaker is tripped (meaning rerank should be skipped)."""
        if len(self._results) < 10:
            return False # not enough data to trip
            
        timeouts = sum(1 for r in self._results if r == "timeout")
        errors = sum(1 for r in self._results if r == "error")
        total = len(self._results)
        
        if (timeouts / total) > self.max_timeout_rate:
            return True
        if (errors / total) > self.max_error_rate:
            return True
            
        return False
        
    @property
    def state(self) -> str:
        return "open" if self.is_open else "closed"


class RerankPolicy:
    """Production policy engine for conditional Cross-Encoder reranking."""
    
    def __init__(self, config_path: str = None):
        if not config_path:
            config_path = str(Path(__file__).parent / "rerank_policy.yaml")
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["rerank_policy"]
            
        guardrails = self.config.get("guardrails", {})
        self.circuit_breaker = CircuitBreaker(
            max_timeout_rate=guardrails.get("max_timeout_rate", 0.01),
            max_error_rate=guardrails.get("max_error_rate", 0.005)
        )
        
        self.budget_tracker = collections.defaultdict(lambda: collections.deque(maxlen=50))
        self.shadow_config = self.config.get("shadow_mode", {})
        self.shadow_mode_enabled = self.shadow_config.get("enabled_initially", False)
        
        self.HARD_SKIP_REASONS = set(self.shadow_config.get("hard_skip_reasons", [
             "service_unhealthy", "budget_exceeded", "circuit_breaker_open", "policy_off", "policy_dense_only"
        ]))
        self.SOFT_SKIP_REASONS = set(self.shadow_config.get("soft_skip_reasons", [
             "family_not_allowed", "insufficient_candidates"
        ]))

    def should_shadow_execute(self, skip_reason: str) -> bool:
        return self.shadow_mode_enabled and skip_reason in self.SOFT_SKIP_REASONS
        
    def record_latency(self, family: str, latency_delta_ms: float):
        """Record the latency delta. Used for budget monitoring."""
        self.budget_tracker[family].append(latency_delta_ms)
        
    def exceeds_budget(self, family: str) -> bool:
        """Check if family's historical latency exceeds configured budget."""
        history = self.budget_tracker[family]
        if len(history) < 10:
            return False # Not enough data
            
        sorted_lat = sorted(history)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        
        budget = self.config.get("family_latency_budgets", {}).get(family, {})
        if not budget:
             budget = self.config.get("guardrails", {})
             
        max_p50 = budget.get("max_p50_delta_ms", 25)
        max_p95 = budget.get("max_p95_delta_ms", 60)
        
        if p50 > max_p50 or p95 > max_p95:
             logger.warning(f"Budget exceeded for {family}: p50={p50}, p95={p95}")
             return True
             
        return False

    def is_eligible(self, family: str, candidate_count: int, service_healthy: bool) -> Dict[str, Any]:
        """
        Calculates if a query is eligible for reranking based on the full policy graph.
        Returns a dictionary containing "eligible": bool, and a "skip_reason": str.
        """
        if self.config.get("mode") == "off" and not self.shadow_mode_enabled:
            return {"eligible": False, "skip_reason": "policy_off"}
            
        if self.config.get("mode") == "dense_only":
            return {"eligible": False, "skip_reason": "policy_dense_only"}

        if family not in self.config.get("enabled_query_families", []):
             return {"eligible": False, "skip_reason": "family_not_allowed"}
             
        if not service_healthy:
             return {"eligible": False, "skip_reason": "service_unhealthy"}
             
        if candidate_count < self.config.get("minimum_candidate_pool", 20):
             return {"eligible": False, "skip_reason": "insufficient_candidates"}
             
        if self.exceeds_budget(family):
             return {"eligible": False, "skip_reason": "budget_exceeded"}
             
        if self.circuit_breaker.is_open:
             return {"eligible": False, "skip_reason": "circuit_breaker_open"}
             
        return {"eligible": True, "skip_reason": None}

    def get_depth(self, family: str) -> int:
        return self.config.get("depth_by_family", {}).get(family, 20)
