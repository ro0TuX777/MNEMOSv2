import pytest
from mnemos.retrieval.policies.rerank_policy import RerankPolicy

class MockRerankPolicy(RerankPolicy):
    def __init__(self, mode="conditional", shadow_enabled=True):
        self.config = {
            "mode": mode,
            "enabled_query_families": ["code_behavior", "why_how"],
            "shadow_mode": {
                "enabled_initially": shadow_enabled,
                "hard_skip_reasons": ["service_unhealthy", "budget_exceeded", "circuit_breaker_open"],
                "soft_skip_reasons": ["family_not_allowed", "insufficient_candidates"]
            },
            "minimum_candidate_pool": 20
        }
        self.shadow_config = self.config["shadow_mode"]
        self.shadow_mode_enabled = self.shadow_config["enabled_initially"]
        self.HARD_SKIP_REASONS = set(self.shadow_config["hard_skip_reasons"])
        self.SOFT_SKIP_REASONS = set(self.shadow_config["soft_skip_reasons"])
        
        from mnemos.retrieval.policies.rerank_policy import CircuitBreaker
        self.circuit_breaker = CircuitBreaker(0.01, 0.005)
        import collections
        self.budget_tracker = collections.defaultdict(lambda: collections.deque(maxlen=50))

def test_shadow_soft_skip():
    policy = MockRerankPolicy()
    # Test soft skip: family_not_allowed
    el = policy.is_eligible("factoid", 100, True)
    assert not el["eligible"]
    assert el["skip_reason"] == "family_not_allowed"
    assert policy.should_shadow_execute(el["skip_reason"]) is True

def test_shadow_hard_skip_health():
    policy = MockRerankPolicy()
    el = policy.is_eligible("code_behavior", 100, False)
    assert not el["eligible"]
    assert el["skip_reason"] == "service_unhealthy"
    assert policy.should_shadow_execute(el["skip_reason"]) is False

def test_shadow_hard_skip_circuit():
    policy = MockRerankPolicy()
    
    # Trip circuit
    for _ in range(15): policy.circuit_breaker.record_timeout()
    
    el = policy.is_eligible("code_behavior", 100, True)
    assert not el["eligible"]
    assert el["skip_reason"] == "circuit_breaker_open"
    assert policy.should_shadow_execute(el["skip_reason"]) is False

def test_shadow_disabled():
    policy = MockRerankPolicy(shadow_enabled=False)
    el = policy.is_eligible("factoid", 100, True)
    assert not el["eligible"]
    assert policy.should_shadow_execute(el["skip_reason"]) is False
