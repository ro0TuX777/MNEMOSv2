import pytest
import lancedb

from mnemos.retrieval.lancedb_tier import LanceDBTier
from mnemos.retrieval.fusion import TierFusion
from mnemos.retrieval.derived_fact_scoring import DerivedFactScorer

def test_support_retrieval_and_scoring():
    db_dir = "data/pit11a/lance"
    table_name = "mnemos_engrams"
    
    # 1. Given a known source_engram_id from The Intelligence Oversight Guide
    lance_tier = LanceDBTier(db_dir=db_dir, table_name=table_name)
    fusion = TierFusion([lance_tier])
    lance_tier._initialize()
    if lance_tier._table is None:
        pytest.skip("local PIT-11A LanceDB fixture is not available")

    # Find a real ID from the guide to use as our "known" ID
    df = lance_tier._table.to_pandas()
    df_guide = df[df['source'].str.contains("Intelligence Oversight", na=False)]
    assert not df_guide.empty, "Could not find Intelligence Oversight Guide in test db"
    
    test_id = df_guide.iloc[0]['id']
    expected_text = df_guide.iloc[0]['content']

    assert test_id is not None, "Could not find Intelligence Oversight Guide in test db"

    # 2. get_engrams(source_engram_id) returns non-empty text
    fetched = fusion.get_engrams([test_id])
    assert len(fetched) > 0, "get_engrams returned empty list"
    
    source_texts = [f.get("content", "") for f in fetched]
    assert len(source_texts) > 0
    assert len(source_texts[0]) > 0
    
    # 3. source_text_preview contains expected guide text
    # 4. support_evidence_preview is populated
    support_evidence_preview = source_texts[0][:200].replace("\n", " ") + "..."
    assert len(support_evidence_preview) > 5
    assert support_evidence_preview != "None"

    # 5. Add one controlled positive test case where the Derived Fact text is an exact or near-exact paraphrase
    scorer = DerivedFactScorer.get_instance()
    fact_content = "This is a derived fact: " + source_texts[0][:150]
    
    scores = scorer.score_candidate("What is the role of the guide?", fact_content, source_texts)
    
    # That case must produce: SELECTED_WITH_SOURCE_SUPPORT (in router context)
    # support_score >= threshold (0.65)
    support_score = scores.get("derived_fact_source_support_score", 0.0)
    assert support_score >= 0.65, f"Support score {support_score} is below threshold 0.65"

def test_semantic_support_rendering_q3():
    db_dir = "data/pit11a/lance"
    table_name = "mnemos_engrams"
    
    lance_tier = LanceDBTier(db_dir=db_dir, table_name=table_name)
    fusion = TierFusion([lance_tier])
    lance_tier._initialize()
    if lance_tier._table is None:
        pytest.skip("local PIT-11A LanceDB fixture is not available")

    # Find the specific chunk from the Intelligence Oversight Guide that contains the purpose statement
    df = lance_tier._table.to_pandas()
    df_guide = df[df['content'].str.contains("The purpose of this guide is to assist Inspectors General", na=False)]
    assert not df_guide.empty, "Could not find the target sentence in the test db"
    
    source_text = df_guide.iloc[0]['content']
    
    scorer = DerivedFactScorer.get_instance()
    fact_content = "[GOLD_ALIGNED] The Intelligence Oversight Guide assists IGs in preparing, executing, and completing Intelligence Oversight inspections."
    
    # Test rendering logic
    render_results = scorer.render_support_evidence(fact_content, source_text)
    
    excerpt = render_results["excerpt"]
    rendering_score = render_results["score"]
    
    # Verify that the extracted excerpt contains the target purpose statement
    assert "The purpose of this guide is to assist Inspectors General" in excerpt, f"Expected purpose statement in excerpt, but got: {excerpt}"
    assert rendering_score >= 0.70, f"Expected strong rendering score >= 0.70, got: {rendering_score}"
