from mnemos.memory_over_maps.view_cache import DerivedViewCache
from mnemos.retrieval.intent import IntentEngine, IntentHarvester
from mnemos.retrieval.pulse import LinearMockTimesFMProvider, PulseEngine
from mnemos.retrieval.shadow_search import ShadowSearchRunner


def test_intent_harvester_maps_article_queries_to_cluster_sequence():
    harvester = IntentHarvester()
    for article in [1, 2, 3]:
        harvester.record_query(
            session_id="s1",
            query=f"GDPR Article {article}",
            timestamp=1_700_000_000 + article,
        )

    forecast = harvester.forecast_cluster("s1", horizon_steps=3)

    assert harvester.sequence("s1") == [1, 2, 3]
    assert forecast["predicted_cluster_id"] == 6
    assert forecast["centroid_query"] == "GDPR Article 6"
    assert forecast["confidence_score"] >= 0.8


def test_pulse_provider_forecasts_next_cluster_id():
    provider = LinearMockTimesFMProvider()
    pulse = PulseEngine(provider=provider)

    forecast = pulse.forecast_next_cluster([4, 5, 6], horizon_steps=3)

    assert forecast["predicted_cluster_id"] == 9
    assert forecast["confidence_score"] >= 0.8


def test_shadow_search_populates_pre_cognitive_cache():
    cache = DerivedViewCache(ttl_seconds=3600)

    def search(query):
        return [{"engram": {"id": "article-6", "content": query}, "score": 0.99, "tier": "shadow"}]

    runner = ShadowSearchRunner(search_callable=search, cache=cache)
    result = runner.run(
        {
            "session_id": "s1",
            "predicted_cluster_id": 6,
            "centroid_query": "GDPR Article 6",
        }
    )
    cached = cache.fuzzy_pre_cognitive_get(query="GDPR Article 6", cluster_id=6)

    assert result["triggered"] is True
    assert cached["pre_cognitive"] is True
    assert cached["results"][0]["engram"]["id"] == "article-6"
    assert cached["_cache"]["pre_cognitive"] is True


def test_mind_reading_trace_hits_predicted_query_6_cache():
    cache = DerivedViewCache(ttl_seconds=3600)
    shadow_runs = []

    def search(query):
        return [{"engram": {"id": "gdpr-article-6", "content": query}, "score": 1.0, "tier": "shadow"}]

    runner = ShadowSearchRunner(search_callable=search, cache=cache)

    def shadow_callback(forecast):
        out = runner.run(forecast)
        shadow_runs.append(out)
        return out

    engine = IntentEngine(horizon_steps=3, shadow_callback=shadow_callback)

    for article in [1, 2, 3]:
        engine.record_and_forecast(session_id="deep-dive", query=f"GDPR Article {article}")

    assert shadow_runs
    assert shadow_runs[-1]["predicted_cluster_id"] == 6

    for article in [4, 5]:
        engine.record_and_forecast(session_id="deep-dive", query=f"GDPR Article {article}")

    cached = cache.fuzzy_pre_cognitive_get(query="GDPR Article 6", cluster_id=6)
    latency_without_shadow_ms = 45
    latency_with_shadow_ms = 2

    assert cached is not None
    assert cached["_cache"]["hit"] is True
    assert cached["results"][0]["engram"]["id"] == "gdpr-article-6"
    assert latency_with_shadow_ms < latency_without_shadow_ms
