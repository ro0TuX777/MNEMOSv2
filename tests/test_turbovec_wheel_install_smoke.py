import pytest

def test_turbovec_import():
    """Verify that turbovec is correctly installed and accessible."""
    try:
        import turbovec
    except ImportError as e:
        pytest.fail(f"Failed to import turbovec: {e}")

def test_turbovec_real_adapter_contract():
    """Verify that the real index adapter initializes without segfaulting or raising NotImplementedError."""
    from mnemos.retrieval.turbovec_tier import RealTurbovecIndexAdapter
    
    dim = 768
    bit_width = 4
    
    # Initialization
    adapter = RealTurbovecIndexAdapter(dim, bit_width)
    assert adapter is not None
    assert adapter.dim == dim
    
    # Adding vectors
    ids = [1, 2, 3]
    vectors = [
        [0.1] * dim,
        [0.2] * dim,
        [0.3] * dim
    ]
    
    try:
        adapter.add(ids, vectors)
    except Exception as e:
        pytest.fail(f"adapter.add failed: {e}")
        
    # Querying
    try:
        q = [0.1] * dim
        results = adapter.search(q, top_k=2)
        assert len(results) <= 2
    except Exception as e:
        pytest.fail(f"adapter.search failed: {e}")
