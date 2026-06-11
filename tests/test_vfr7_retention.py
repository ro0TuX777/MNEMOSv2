import os
import time
import tempfile
import pytest
from tools.purge_sidecar_evaluations import purge_old_exports

def test_gate_7_retention_enforcement():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an expired file (8 days old)
        expired_path = os.path.join(tmpdir, "expired.json")
        with open(expired_path, "w") as f:
            f.write("{}")
            
        # Create a fresh file (1 day old)
        fresh_path = os.path.join(tmpdir, "fresh.json")
        with open(fresh_path, "w") as f:
            f.write("{}")
            
        now = time.time()
        os.utime(expired_path, (now - 8 * 86400, now - 8 * 86400))
        os.utime(fresh_path, (now - 1 * 86400, now - 1 * 86400))
        
        # Act
        count = purge_old_exports(tmp_dir=tmpdir, days=7)
        
        # Assert
        assert count == 1
        assert not os.path.exists(expired_path)
        assert os.path.exists(fresh_path)
