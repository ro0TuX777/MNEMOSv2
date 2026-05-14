import os
import json
import time
import hashlib
from datetime import datetime

class ShadowReceipts:
    @staticmethod
    def emit_telemetry(payload: dict, output_dir: str):
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            event_id = hashlib.sha256(f"{timestamp}_{payload.get('query_hash', '')}".encode()).hexdigest()[:12]
            
            filepath = os.path.join(output_dir, f"shadow_event_{timestamp}_{event_id}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            # Silent fail for telemetry emission to avoid breaking MNEMOS
            print(f"Shadow receipt emission failed: {e}")
