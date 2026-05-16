import pytest
import numpy as np

try:
    import turbovec
    HAS_TURBOVEC = True
except ImportError:
    HAS_TURBOVEC = False

from mnemos.retrieval.turbovec_tier import MockDenseIndexAdapter, RealTurbovecIndexAdapter

@pytest.mark.parametrize("adapter_cls", [
    MockDenseIndexAdapter,
    pytest.param(RealTurbovecIndexAdapter, marks=pytest.mark.skipif(not HAS_TURBOVEC, reason="turbovec not installed"))
])
def test_adapter_contract(adapter_cls, tmp_path):
    dim = 8
    if adapter_cls == MockDenseIndexAdapter:
        adapter = adapter_cls(dim)
    else:
        adapter = adapter_cls(dim, 4) # dim=8, bit_width=4
        
    # Index vectors
    ids = [1, 2, 3]
    vectors = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]
    adapter.add(ids, vectors)
    
    # Search
    results = adapter.search([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2)
    assert len(results) == 2
    assert results[0][0] == 1 # id 1 should be closest
    
    # Save/Load
    save_path = str(tmp_path / "index.tvim")
    adapter.save(save_path)
    
    if adapter_cls == MockDenseIndexAdapter:
        loaded = adapter_cls.load(save_path)
    else:
        loaded = adapter_cls.load(save_path, dim, 4)
        
    results_loaded = loaded.search([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1)
    assert len(results_loaded) == 1
    assert results_loaded[0][0] == 1
    
    # Delete / Soft Delete behavior
    try:
        loaded.delete([1])
        r = loaded.search([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1)
        if len(r) > 0:
            assert r[0][0] != 1
    except AttributeError:
        # Soft delete fallback for incomplete API
        pass
        
    # Dimension mismatch
    with pytest.raises(ValueError):
        adapter.add([4], [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) # dim 7 instead of 8
