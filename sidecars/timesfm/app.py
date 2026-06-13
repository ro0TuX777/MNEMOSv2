"""MNEMOS TimesFM sidecar.

Local-model-first HTTP wrapper for TimesFM 2.5 pulse forecasting.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from flask import Flask, jsonify, request


MODEL_PATH = os.getenv("TIMESFM_MODEL_PATH", "/models/timesfm-2.5-200m-pytorch")
MODEL_ID = os.getenv("TIMESFM_MODEL_ID", "google/timesfm-2.5-200m-pytorch")
ALLOW_HF_DOWNLOAD = os.getenv("TIMESFM_ALLOW_HF_DOWNLOAD", "false").lower() in {
    "true",
    "1",
    "yes",
}
MAX_CONTEXT = int(os.getenv("TIMESFM_MAX_CONTEXT", "1024"))
MAX_HORIZON = int(os.getenv("TIMESFM_MAX_HORIZON", "256"))
PER_CORE_BATCH_SIZE = int(os.getenv("TIMESFM_BATCH_SIZE", "4"))

METRICS = ["query_count", "p95_latency_ms", "cache_hit_rate", "degrade_count"]

app = Flask(__name__)
_model = None
_model_error = None
_model_lock = threading.Lock()


def _extract_value(raw: Any) -> float:
    if isinstance(raw, dict):
        raw = raw.get("point", raw.get("q50", 0.0))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _load_model():
    global _model, _model_error
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            import torch
            import timesfm

            torch.set_float32_matmul_precision("high")
            model_source = MODEL_PATH
            if not Path(MODEL_PATH).exists():
                if not ALLOW_HF_DOWNLOAD:
                    raise FileNotFoundError(
                        f"TimesFM model path not found: {MODEL_PATH}. "
                        "Mount pinned artifacts or set TIMESFM_ALLOW_HF_DOWNLOAD=true."
                    )
                model_source = MODEL_ID

            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_source)
            model.compile(
                timesfm.ForecastConfig(
                    max_context=MAX_CONTEXT,
                    max_horizon=MAX_HORIZON,
                    normalize_inputs=True,
                    per_core_batch_size=PER_CORE_BATCH_SIZE,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            _model = model
            _model_error = None
            return _model
        except Exception as exc:
            _model_error = str(exc)
            raise


def _series_from_patches(patches: List[Dict[str, Any]]) -> List[np.ndarray]:
    series = []
    for metric in METRICS:
        values = [_extract_value(patch.get(metric, 0.0)) for patch in patches]
        series.append(np.asarray(values or [0.0], dtype=np.float32))
    return series


def _metric_value(metric: str, value: float) -> float:
    if metric in {"query_count", "degrade_count"}:
        return max(0.0, value)
    if metric == "cache_hit_rate":
        return min(1.0, max(0.0, value))
    return max(0.0, value)


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "healthy" if _model_error is None else "degraded",
            "service": "mnemos-timesfm",
            "model_path": MODEL_PATH,
            "model_loaded": _model is not None,
            "error": _model_error,
        }
    )


@app.post("/forecast")
def forecast():
    started = time.perf_counter()
    body = request.get_json(silent=True) or {}
    patches = body.get("patches", [])
    horizon = int(body.get("horizon_minutes", 15))
    if not isinstance(patches, list) or not patches:
        return jsonify({"error": "patches must be a non-empty list"}), 400
    if horizon < 1 or horizon > MAX_HORIZON:
        return jsonify({"error": f"horizon_minutes must be between 1 and {MAX_HORIZON}"}), 400

    try:
        model = _load_model()
        point, quantiles = model.forecast(
            horizon=horizon,
            inputs=_series_from_patches(patches),
        )
    except Exception as exc:
        return jsonify({"error": str(exc), "provider": "timesfm_sidecar"}), 503

    last_bucket = int(patches[-1].get("bucket_start", int(time.time())))
    rows = []
    for step in range(horizon):
        bucket = last_bucket + (step + 1) * 60
        row: Dict[str, Any] = {
            "bucket_start": bucket,
            "bucket_start_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(bucket)),
        }
        for metric_idx, metric in enumerate(METRICS):
            value = _metric_value(metric, float(point[metric_idx, step]))
            q10 = _metric_value(metric, float(quantiles[metric_idx, step, 1]))
            q90 = _metric_value(metric, float(quantiles[metric_idx, step, 9]))
            row[metric] = {
                "point": round(value, 4),
                "q10": round(min(q10, q90, value), 4),
                "q50": round(value, 4),
                "q90": round(max(q10, q90, value), 4),
            }
        rows.append(row)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return jsonify(
        {
            "provider": "timesfm_sidecar",
            "mode": "timesfm",
            "model": "timesfm-2.5-200m-pytorch",
            "horizon_minutes": horizon,
            "targets": METRICS,
            "confidence_score": 0.75,
            "inference_ms": round(elapsed_ms, 3),
            "patches": rows,
        }
    )


@app.post("/forecast_intent")
def forecast_intent():
    body = request.get_json(silent=True) or {}
    sequence = body.get("sequence", [])
    horizon_steps = int(body.get("horizon_steps", 3))
    if not isinstance(sequence, list) or not sequence:
        return jsonify({"error": "sequence must be a non-empty list"}), 400
    try:
        seq = [int(x) for x in sequence]
    except (TypeError, ValueError):
        return jsonify({"error": "sequence must contain integers"}), 400

    # Cluster IDs are discrete. Until a dedicated intent head exists, use the
    # TimesFM sidecar boundary with a deterministic trend projection.
    if len(seq) == 1:
        predicted = seq[-1]
        confidence = 0.35
    else:
        deltas = [b - a for a, b in zip(seq[:-1], seq[1:])]
        window = deltas[-3:]
        step = round(sum(window) / len(window))
        predicted = int(seq[-1] + step * max(1, horizon_steps))
        confidence = 0.88 if len(set(window)) == 1 else 0.62

    return jsonify(
        {
            "provider": "timesfm_sidecar",
            "mode": "timesfm_intent_projection",
            "history": seq,
            "predicted_cluster_id": predicted,
            "horizon_steps": horizon_steps,
            "confidence_score": confidence,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8711")))
