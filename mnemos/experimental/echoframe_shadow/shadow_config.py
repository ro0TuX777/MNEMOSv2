import os

class ShadowConfig:
    """Configuration for EchoFrame Shadow Runtime."""
    
    @classmethod
    def is_enabled(cls) -> bool:
        return os.environ.get("MNEMOS_ECHOFRAME_SHADOW_ENABLED", "false").lower() == "true"
        
    @classmethod
    def get_mode(cls) -> str:
        return os.environ.get("MNEMOS_ECHOFRAME_SHADOW_MODE", "compact_semantic_minEvidence_hysteresis_v0")
        
    @classmethod
    def get_sample_rate(cls) -> float:
        try:
            return float(os.environ.get("MNEMOS_ECHOFRAME_SHADOW_SAMPLE_RATE", "1.0"))
        except ValueError:
            return 1.0
            
    @classmethod
    def get_output_dir(cls) -> str:
        return os.environ.get("MNEMOS_ECHOFRAME_SHADOW_OUTPUT_DIR", "runtime/echoframe_shadow/")
        
    @classmethod
    def get_fail_closed(cls) -> bool:
        return os.environ.get("MNEMOS_ECHOFRAME_SHADOW_FAIL_CLOSED", "false").lower() == "true"
        
class PilotConfig:
    """Configuration for EchoFrame LLM-Facing Pilot."""
    
    @classmethod
    def is_enabled(cls) -> bool:
        return os.environ.get("MNEMOS_ECHOFRAME_LLM_FACING_ENABLED", "false").lower() == "true"
        
    @classmethod
    def get_mode(cls) -> str:
        return os.environ.get("MNEMOS_ECHOFRAME_LLM_FACING_MODE", "compact_semantic_minEvidence_hysteresis_v0")
        
    @classmethod
    def get_sample_rate(cls) -> float:
        try:
            return float(os.environ.get("MNEMOS_ECHOFRAME_LLM_FACING_SAMPLE_RATE", "0.05"))
        except ValueError:
            return 0.05
            
    @classmethod
    def get_fail_closed(cls) -> bool:
        return os.environ.get("MNEMOS_ECHOFRAME_LLM_FACING_FAIL_CLOSED", "true").lower() == "true"
        
    @classmethod
    def get_require_validation(cls) -> bool:
        return os.environ.get("MNEMOS_ECHOFRAME_LLM_FACING_REQUIRE_VALIDATION", "true").lower() == "true"
        
    @classmethod
    def get_allow_high_risk(cls) -> bool:
        return os.environ.get("MNEMOS_ECHOFRAME_LLM_FACING_ALLOW_HIGH_RISK", "false").lower() == "true"
        
    @classmethod
    def get_output_dir(cls) -> str:
        return os.environ.get("MNEMOS_ECHOFRAME_LLM_FACING_OUTPUT_DIR", "runtime/echoframe_pilot/")

class DefaultOnConfig:
    """Configuration for EchoFrame Default-On Eligible mode (Phase 7)."""
    
    @classmethod
    def is_enabled(cls) -> bool:
        return os.environ.get("MNEMOS_ECHOFRAME_DEFAULT_ON_ELIGIBLE", "true").lower() == "true"
        
    @classmethod
    def is_kill_switch_active(cls) -> bool:
        return os.environ.get("MNEMOS_ECHOFRAME_KILL_SWITCH", "false").lower() == "true"
        
    @classmethod
    def get_mode(cls) -> str:
        return os.environ.get("MNEMOS_ECHOFRAME_MODE", "compact_semantic_minEvidence_hysteresis_v0")
        
    @classmethod
    def get_fail_closed(cls) -> bool:
        return os.environ.get("MNEMOS_ECHOFRAME_FAIL_CLOSED", "true").lower() == "true"
        
    @classmethod
    def get_require_validation(cls) -> bool:
        return os.environ.get("MNEMOS_ECHOFRAME_REQUIRE_VALIDATION", "true").lower() == "true"
        
    @classmethod
    def get_allow_high_risk(cls) -> bool:
        return os.environ.get("MNEMOS_ECHOFRAME_ALLOW_HIGH_RISK", "false").lower() == "true"
        
    @classmethod
    def get_output_dir(cls) -> str:
        return os.environ.get("MNEMOS_ECHOFRAME_OUTPUT_DIR", "runtime/echoframe_default/")
