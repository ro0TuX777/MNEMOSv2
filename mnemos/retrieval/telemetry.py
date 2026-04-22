"""
Retrieval Telemetry Sink
========================

Handles durable persistence of retrieval decisions, enabling ops 
observability, shadow reporting, and precision/recall analysis.
"""

import json
import logging
from typing import Dict, Any
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

logger = logging.getLogger(__name__)

class RetrievalTelemetrySink:
    """Abstract interface for emitting retrieval telemetry."""
    def emit(self, event: Dict[str, Any]) -> None:
        raise NotImplementedError

class JsonlTelemetrySink(RetrievalTelemetrySink):
    """
    Appends retrieval telemetry events to a rotating JSONL file constraint.
    """
    def __init__(self, filepath: str = "logs/retrieval_telemetry.jsonl"):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("mnemos.retrieval.telemetry.jsonl")
        self.logger.setLevel(logging.INFO)
        # Prevent propagation to root logger to avoid console spam
        self.logger.propagate = False
        
        if not self.logger.handlers:
            # Rotate daily at midnight, keeping 30 days
            handler = TimedRotatingFileHandler(
                filename=str(path),
                when="midnight",
                interval=1,
                backupCount=30,
                encoding="utf-8"
            )
            # Explicitly format as raw message (which will be JSON)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
            
    def emit(self, event: Dict[str, Any]) -> None:
        try:
            self.logger.info(json.dumps(event))
        except Exception as e:
            logger.error(f"Failed to emit JSONL telemetry: {e}")

class PostgresTelemetrySink(RetrievalTelemetrySink):
    def __init__(self, dsn: str):
        self.dsn = dsn
    def emit(self, event: Dict[str, Any]) -> None:
        # Placeholder for PostgreSQL implementation
        pass

def get_telemetry_sink(config: Dict[str, Any] = None) -> RetrievalTelemetrySink:
    if not config:
        config = {}
    
    sink_type = config.get("sink", "jsonl")
    
    if sink_type == "jsonl":
        path = config.get("jsonl_path", "logs/retrieval_telemetry.jsonl")
        return JsonlTelemetrySink(path)
    elif sink_type == "postgres":
         return PostgresTelemetrySink(config.get("postgres_dsn", ""))
    
    # fallback
    return JsonlTelemetrySink("logs/retrieval_telemetry.jsonl")
