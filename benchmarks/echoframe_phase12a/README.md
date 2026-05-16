# EchoFrame Phase 12A Placement Benchmark

This module provides a benchmark suite for testing different packet layouts in EchoFrame to measure their impact on LLM reasoning fidelity (Lost in the Middle).

## Layouts Tested
- A: Baseline
- B: Governance-first
- C: Governance-recap
- D: Top-and-bottom constraints
- E: Source-local

## Running
```bash
python benchmarks/echoframe_phase12a/run_phase12a_placement_benchmark.py
```
