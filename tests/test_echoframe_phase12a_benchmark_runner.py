import sys
import os
import subprocess

def test_runner():
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), '../benchmarks/echoframe_phase12a/run_phase12a_placement_benchmark.py'),
        "--layouts", "A",
        "--out", "benchmarks/outputs/raw"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Benchmark complete" in res.stdout
