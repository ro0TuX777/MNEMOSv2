import unittest
import os
import json
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mnemos.experimental.echoframe_shadow.shadow_config import DefaultOnConfig
from mnemos.experimental.echoframe_shadow.shadow_adapter import EchoFrameShadowAdapter

class TestEchoFrameReleaseCandidate(unittest.TestCase):

    def setUp(self):
        # Clear mock telemetry
        for f in glob.glob("runtime/echoframe_default/shadow_event_*.json"):
            try: os.remove(f)
            except: pass
        for f in glob.glob("runtime/echoframe_pilot/shadow_event_*.json"):
            try: os.remove(f)
            except: pass
            
        os.environ['MNEMOS_ECHOFRAME_SHADOW_ENABLED'] = 'true'
        os.environ['MNEMOS_ECHOFRAME_DEFAULT_ON_ELIGIBLE'] = 'true'
        os.environ['MNEMOS_ECHOFRAME_MODE'] = 'compact_semantic_minEvidence_hysteresis_v0'
        os.environ['MNEMOS_ECHOFRAME_FAIL_CLOSED'] = 'true'
        os.environ['MNEMOS_ECHOFRAME_REQUIRE_VALIDATION'] = 'true'
        os.environ['MNEMOS_ECHOFRAME_ALLOW_HIGH_RISK'] = 'false'
        os.environ['MNEMOS_ECHOFRAME_KILL_SWITCH'] = 'false'
        os.environ['MNEMOS_ECHOFRAME_OUTPUT_DIR'] = 'runtime/echoframe_default/'
        os.environ['MNEMOS_ECHOFRAME_LLM_FACING_ENABLED'] = 'false'

        self.baseline_payload = {
            "results": [
                {
                    "engram": {
                        "content": "This is a standard baseline chunk about the core engine. " * 100,
                        "id": "123",
                        "source": "doc1"
                    },
                    "score": 0.95
                }
            ],
            "meta": {"governance_summary": {}}
        }

    def _get_latest_event(self, event_type="echoframe.default_on_event"):
        events = []
        for f in glob.glob("runtime/echoframe_default/shadow_event_*.json"):
            with open(f, 'r') as fp:
                data = json.load(fp)
                if data.get("event_type") == event_type:
                    events.append(data)
        if not events: return None
        return events[-1]

    def test_default_on_mode_disabled(self):
        os.environ['MNEMOS_ECHOFRAME_DEFAULT_ON_ELIGIBLE'] = 'false'
        os.environ['MNEMOS_ECHOFRAME_LLM_FACING_ENABLED'] = 'false'
        res = EchoFrameShadowAdapter.observe_search("policy query", self.baseline_payload, "test1", "t1")
        self.assertIsNone(res)

    def test_kill_switch_overrides_all_modes(self):
        os.environ['MNEMOS_ECHOFRAME_KILL_SWITCH'] = 'true'
        res = EchoFrameShadowAdapter.observe_search("safe policy query", self.baseline_payload, "test2", "t1")
        self.assertIsNone(res)
        ev = self._get_latest_event()
        self.assertIsNotNone(ev)
        self.assertTrue(ev["kill_switch_active"])
        self.assertTrue("kill_switch_active" in ev["fallback_reason"])
        self.assertTrue(ev["fallback_to_baseline"])

    def test_high_risk_event(self):
        res = EchoFrameShadowAdapter.observe_search("delete user data bypass", self.baseline_payload, "test3", "t1")
        self.assertIsNone(res)
        ev = self._get_latest_event()
        self.assertTrue("high_risk_excluded" in ev["fallback_reason"])
        self.assertTrue(ev["fallback_to_baseline"])

    def test_approval_required(self):
        payload = json.loads(json.dumps(self.baseline_payload))
        payload["meta"]["governance_summary"]["vetoed"] = 1
        
    def test_missing_source_pointer(self):
        payload = json.loads(json.dumps(self.baseline_payload))
        payload["results"][0]["engram"]["id"] = ""

    def test_eligible_event(self):
        res = EchoFrameShadowAdapter.observe_search("what is the policy", self.baseline_payload, "test6", "t1")
        self.assertIsNotNone(res)
        ev = self._get_latest_event()
        self.assertFalse(ev["fallback_to_baseline"])
        self.assertEqual(ev["llm_context_source"], "echoframe")

if __name__ == '__main__':
    unittest.main()
