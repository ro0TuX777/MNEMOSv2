from mnemos.runtime.profile_selector import resolve_profile, ProfileConfig

def get_retrieval_tier():
    """
    Factory function that instantiates the retrieval tier based on the resolved profile.
    """
    config = resolve_profile()
    
    if config.profile_name == "portable_memory_appliance":
        # Import dynamically so we don't blow up if turbovec isn't installed and we aren't using it
        try:
            from mnemos.retrieval.turbovec_config import TurbovecConfig
            from mnemos.retrieval.turbovec_tier import TurbovecTier
            from mnemos.retrieval.turbovec_fusion import TurbovecFusion
            
            # Using default config path for runtime tier; can be overridden by injection
            tc = TurbovecConfig(embedding_dim=768, bit_width=4, storage_path="runtime/turbovec_storage")
            tier = TurbovecTier(tc, use_mock=False)
            return TurbovecFusion(tier)
            
        except ImportError as e:
            raise RuntimeError(f"Cannot initialize portable memory appliance: {e}")
            
    elif config.profile_name == "core_memory_appliance":
        # Stub for Qdrant tier injection
        return _MockQdrantTier()
        
    elif config.profile_name == "governance_native":
        # Stub for pgvector tier injection
        return _MockPgvectorTier()
        
    else:
        raise ValueError(f"Unknown profile: {config.profile_name}")

class _MockQdrantTier:
    def __init__(self):
        self.name = "qdrant"

class _MockPgvectorTier:
    def __init__(self):
        self.name = "pgvector"
