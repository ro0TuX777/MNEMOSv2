import sys
import os
import subprocess

def test_phase12b_runner_mock():
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), '../benchmarks/echoframe_phase12b/run_phase12b_format_benchmark.py'),
        "--formats", "1_baseline,2_layout_d",
        "--out", "benchmarks/outputs/raw_test"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Benchmark complete" in res.stdout
