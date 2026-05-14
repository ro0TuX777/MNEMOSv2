import os
import sys
import unittest
import json

# Append path to import MNEMOS app
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from mnemos.experimental.echoframe_shadow.shadow_config import ShadowConfig

# To test properly we mock MNEMOS payload
class TestShadowIntegration(unittest.TestCase):

    def setUp(self):
        # Reset env
        if "MNEMOS_ECHOFRAME_SHADOW_ENABLED" in os.environ:
            del os.environ["MNEMOS_ECHOFRAME_SHADOW_ENABLED"]
        if "MNEMOS_ECHOFRAME_SHADOW_OUTPUT_DIR" in os.environ:
            del os.environ["MNEMOS_ECHOFRAME_SHADOW_OUTPUT_DIR"]
            
        self.mock_payload = {
            "results": [
                {
                    "engram": {
                        "id": "E1",
                        "source": "S1",
                        "content": "This is a mock MNEMOS payload for exact threshold 5.0 tests."
                    }
                }
            ],
            "meta": {
                "governance_summary": {}
            }
        }
        
    def test_shadow_mode_disabled_by_default(self):
        self.assertFalse(ShadowConfig.is_enabled())

    def test_enabling_shadow_does_not_alter_baseline(self):
        os.environ["MNEMOS_ECHOFRAME_SHADOW_ENABLED"] = "true"
        os.environ["MNEMOS_ECHOFRAME_SHADOW_OUTPUT_DIR"] = "test_output_dir"
        
        from mnemos.experimental.echoframe_shadow import EchoFrameShadowAdapter
        
        # Make a copy to check for mutation
        payload_copy = json.loads(json.dumps(self.mock_payload))
        
        EchoFrameShadowAdapter.observe_search("What is the exact threshold?", payload_copy)
        
        self.assertEqual(payload_copy, self.mock_payload)
        self.assertTrue(os.path.exists("test_output_dir"))
        
    def test_shadow_failure_does_not_break_runtime(self):
        os.environ["MNEMOS_ECHOFRAME_SHADOW_ENABLED"] = "true"
        os.environ["MNEMOS_ECHOFRAME_SHADOW_OUTPUT_DIR"] = "X:/invalid/path/that/will/fail"
        
        from mnemos.experimental.echoframe_shadow import EchoFrameShadowAdapter
        
        # Should fail silently
        EchoFrameShadowAdapter.observe_search("What is the exact threshold?", self.mock_payload)
        
    def test_llm_context_modified_is_false(self):
        os.environ["MNEMOS_ECHOFRAME_SHADOW_ENABLED"] = "true"
        os.environ["MNEMOS_ECHOFRAME_SHADOW_OUTPUT_DIR"] = "test_output_dir_2"
        
        from mnemos.experimental.echoframe_shadow import EchoFrameShadowAdapter
        EchoFrameShadowAdapter.observe_search("What is the exact threshold?", self.mock_payload)
        
        import glob
        files = glob.glob("test_output_dir_2/shadow_event_*.json")
        self.assertTrue(len(files) > 0)
        
        with open(files[-1], 'r') as f:
            data = json.load(f)
            
        self.assertFalse(data["llm_context_modified"])
        self.assertIn("token_ratio", data)
        self.assertIn("safety_gate_failures", data)
        
    def test_shadow_exception_emits_failure_telemetry(self):
        os.environ["MNEMOS_ECHOFRAME_SHADOW_ENABLED"] = "true"
        os.environ["MNEMOS_ECHOFRAME_SHADOW_OUTPUT_DIR"] = "test_output_dir_3"
        from mnemos.experimental.echoframe_shadow import EchoFrameShadowAdapter
        
        # Simulate a bad payload
        bad_payload = None
        
        try:
            EchoFrameShadowAdapter.observe_search("Test query", bad_payload)
        except Exception as e:
            EchoFrameShadowAdapter.emit_failure("Test query", e)
            
        import glob
        files = glob.glob("test_output_dir_3/shadow_event_*.json")
        self.assertTrue(len(files) > 0)
        
        with open(files[-1], 'r') as f:
            data = json.load(f)
            
        self.assertEqual(data["event_type"], "echoframe.shadow_packet_failed")
        self.assertFalse(data["llm_context_modified"])
        self.assertTrue(data["baseline_runtime_unaffected"])
        self.assertIn("error_type", data)
        
if __name__ == '__main__':
    unittest.main()
