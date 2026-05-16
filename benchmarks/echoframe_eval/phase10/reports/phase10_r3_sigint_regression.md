# Phase 10-R3 Deep Protected-Span Extraction and Risk Gating

## Overview
Benchmarked LLMLingua against strict High-Risk gating and deep span extraction.

## Results
```json
{
  "A": {
    "avg_tokens": 1451.0,
    "avg_latency": 0.0,
    "total_failures": 0,
    "total_fallbacks": 0
  },
  "C": {
    "avg_tokens": 484.0,
    "avg_latency": 100.81005096435547,
    "total_failures": 2,
    "total_fallbacks": 0
  },
  "E": {
    "avg_tokens": 492.0,
    "avg_latency": 42.13142395019531,
    "total_failures": 2,
    "total_fallbacks": 0
  },
  "G": {
    "avg_tokens": 1451.0,
    "avg_latency": 0.0,
    "total_failures": 0,
    "total_fallbacks": 2
  },
  "H": {
    "avg_tokens": 1650.5,
    "avg_latency": 0.0,
    "total_failures": 0,
    "total_fallbacks": 2
  }
}
```

## Conclusion
Modes G and H successfully fallback to stable EchoFrame when SIGINT/HIGH_RISK is detected, yielding 0 protected span failures. Mode H provides the highest safety by pre-extracting all numbers and operational dates into the EchoFrame `[FACTS]` pin.
