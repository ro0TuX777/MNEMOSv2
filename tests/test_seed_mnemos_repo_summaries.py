from tools.seed_mnemos_repo_summaries import DEFAULT_PARAPHRASE_QUERIES, DEFAULT_SUMMARIES


def test_default_repo_summaries_cover_gatemem_baseline_and_blocker():
    assert len(DEFAULT_SUMMARIES) >= 6
    joined = " ".join(item["content"] for item in DEFAULT_SUMMARIES)
    assert "GateMem G4 is frozen for regression testing only" in joined
    assert "Further GateMem policy and implementation work is paused" in joined
    assert "GateMem G4 frozen regression baseline" in joined
    assert "No further internal prototyping authorized" in joined


def test_default_paraphrase_queries_cover_alias_forms():
    assert "GateMem G4 frozen regression baseline" in DEFAULT_PARAPHRASE_QUERIES
    assert "no further internal prototyping authorized" in DEFAULT_PARAPHRASE_QUERIES
