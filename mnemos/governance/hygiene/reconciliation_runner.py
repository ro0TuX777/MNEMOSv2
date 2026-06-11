"""
Knowledge reconciliation runner for contradiction clusters.

Phase 10 turns detected contradictions into derived Resolution Engrams.  The
runner can consume an existing ContradictionSweepReport or run the sweep itself
in dry-run mode, then synthesize one consensus engram per conflict cluster.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mnemos.engram.model import Engram
from mnemos.governance.hygiene.contradiction_sweep import (
    ContradictionSweepRecord,
    ContradictionSweepReport,
    ContradictionSweepRunner,
)
from mnemos.governance.models.memory_state import GovernanceMeta


@dataclass
class ReconciliationRecord:
    """Outcome for one synthesized resolution engram."""

    cluster_key: str
    conflict_group_id: str
    parent_ids: List[str]
    resolution_engram: Engram

    def to_dict(self) -> Dict[str, object]:
        return {
            "cluster_key": self.cluster_key,
            "conflict_group_id": self.conflict_group_id,
            "parent_ids": list(self.parent_ids),
            "resolution_engram": self.resolution_engram.to_dict(
                include_governance=True
            ),
        }


@dataclass
class ReconciliationReport:
    """Summary of one reconciliation pass."""

    dry_run: bool
    clusters_scanned: int = 0
    contradictions_found: int = 0
    resolution_engram_writes: int = 0
    skipped: int = 0
    records: List[ReconciliationRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "dry_run": self.dry_run,
            "clusters_scanned": self.clusters_scanned,
            "contradictions_found": self.contradictions_found,
            "resolution_engram_writes": self.resolution_engram_writes,
            "skipped": self.skipped,
            "records": [record.to_dict() for record in self.records],
        }


class ConsensusSynthesizer:
    """SMC-compatible reconciliation synthesizer with deterministic fallback."""

    def synthesize(self, *, record: ContradictionSweepRecord, parents: List[Engram]) -> str:
        prompt = self._prompt(record, parents)
        llm = self._call_smc_llm(prompt)
        if llm:
            return llm.strip()
        return self._fallback(record, parents)

    def _call_smc_llm(self, prompt: str) -> Optional[str]:
        base_url = os.getenv("SMC_LLM_BASE_URL", "").strip()
        model = os.getenv("SMC_LLM_MODEL", "").strip()
        if not base_url or not model:
            return None
        try:
            import requests

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are the MNEMOS Consensus Engine. Synthesize "
                            "only from the supplied memories and preserve provenance."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": float(os.getenv("SMC_LLM_TEMPERATURE", "0")),
                "max_tokens": int(os.getenv("SMC_LLM_MAX_TOKENS", "512")),
            }
            headers = {"Content-Type": "application/json"}
            api_key = os.getenv("SMC_LLM_API_KEY", "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            resp = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=int(os.getenv("SMC_LLM_TIMEOUT_SECONDS", "60")),
            )
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])
        except Exception:
            return None

    @staticmethod
    def _prompt(record: ContradictionSweepRecord, parents: List[Engram]) -> str:
        lines = [
            "You are the MNEMOS Consensus Engine.",
            "Here are conflicting memories. Synthesize a Resolution Engram that",
            "either reconciles the difference or explicitly states the unresolved",
            "conflict with provenance for both.",
            f"CLUSTER: {record.cluster_key}",
            f"CONFLICT_GROUP: {record.conflict_group_id}",
            "PARENT ENGRAMS:",
        ]
        for parent in parents:
            gov = parent.governance
            value = gov.normalized_value if gov else ""
            lines.append(
                f"- {parent.id} value={value!r} source={parent.source!r}: "
                f"{' '.join(parent.content.split())[:700]}"
            )
        return "\n".join(lines)

    @staticmethod
    def _fallback(record: ContradictionSweepRecord, parents: List[Engram]) -> str:
        value_lines = []
        for parent in parents:
            gov = parent.governance
            value = gov.normalized_value if gov else "unknown"
            snippet = " ".join(parent.content.split())[:220]
            value_lines.append(f"{parent.id} says {value}: {snippet}")
        joined = "; ".join(value_lines)
        return (
            f"Resolution for {record.cluster_key}: unresolved conflict retained. "
            f"Conflicting parent memories are {joined}. "
            f"Preferred parent by current governance sweep: {record.winner_id}. "
            "Use this resolution engram as the consensus entry point until a "
            "human or stronger source resolves the conflict."
        )


class ReconciliationRunner:
    """Synthesize Resolution Engrams from contradiction sweep output."""

    def __init__(
        self,
        *,
        synthesizer: Optional[ConsensusSynthesizer] = None,
        sweep_runner: Optional[ContradictionSweepRunner] = None,
    ) -> None:
        self._synthesizer = synthesizer or ConsensusSynthesizer()
        self._sweep_runner = sweep_runner or ContradictionSweepRunner()

    def run(
        self,
        engrams: List[Engram],
        *,
        sweep_report: Optional[ContradictionSweepReport] = None,
        dry_run: bool = True,
        indexer: Optional[object] = None,
    ) -> ReconciliationReport:
        if sweep_report is None:
            sweep_report = self._sweep_runner.run(engrams, dry_run=True)

        engram_by_id = {engram.id: engram for engram in engrams}
        report = ReconciliationReport(
            dry_run=dry_run,
            clusters_scanned=sweep_report.clusters_scanned,
            contradictions_found=sweep_report.contradictions_found,
            skipped=sweep_report.skipped,
        )

        to_write: List[Engram] = []
        for record in sweep_report.records:
            parent_ids = self._parent_ids(record)
            parents = [engram_by_id[eid] for eid in parent_ids if eid in engram_by_id]
            if len(parents) < 2:
                report.skipped += 1
                continue

            resolution = self._build_resolution_engram(record, parents)
            report.records.append(
                ReconciliationRecord(
                    cluster_key=record.cluster_key,
                    conflict_group_id=record.conflict_group_id,
                    parent_ids=parent_ids,
                    resolution_engram=resolution,
                )
            )
            to_write.append(resolution)

        if not dry_run and to_write:
            if indexer is None or not hasattr(indexer, "index"):
                raise ValueError("indexer with index(engrams) is required when dry_run=False")
            written = indexer.index(to_write)
            report.resolution_engram_writes = int(written or 0)

        return report

    def _build_resolution_engram(
        self,
        record: ContradictionSweepRecord,
        parents: List[Engram],
    ) -> Engram:
        entity_key, attribute_key = self._split_cluster_key(record.cluster_key)
        parent_ids = [parent.id for parent in parents]
        content = self._synthesizer.synthesize(record=record, parents=parents)
        source_entity = entity_key.replace(":", "_").replace("/", "_")
        truthset_case_ids = {
            str(parent.metadata.get("truthset_case_id"))
            for parent in parents
            if parent.metadata.get("truthset_case_id")
        }
        metadata = {
            "is_resolution_engram": True,
            "conflict_group_id": record.conflict_group_id,
            "parent_ids": parent_ids,
            "resolution_status": "unresolved_or_reconciled",
        }
        if len(truthset_case_ids) == 1:
            metadata["truthset_case_id"] = next(iter(truthset_case_ids))
        resolution = Engram(
            id=f"resolution_{record.conflict_group_id.replace(':', '_')}",
            content=content,
            source=f"derived://reconciliation/{source_entity}",
            edges=parent_ids,
            metadata=metadata,
            governance=GovernanceMeta(
                memory_type="derived",
                source_type="derived",
                source_id=f"reconciliation:{record.conflict_group_id}",
                derived_from=parent_ids,
                entity_key=entity_key,
                attribute_key=attribute_key,
                normalized_value=f"resolution:{record.conflict_group_id}",
                source_authority=0.75,
                trust_score=0.95,
                utility_score=1.0,
                conflict_group_id=record.conflict_group_id,
                conflict_status="winner",
            ),
        )
        return resolution

    @staticmethod
    def _parent_ids(record: ContradictionSweepRecord) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for eid in [record.winner_id, *record.loser_ids]:
            if eid not in seen:
                seen.add(eid)
                ordered.append(eid)
        return ordered

    @staticmethod
    def _split_cluster_key(cluster_key: str) -> tuple[str, str]:
        if ":" not in cluster_key:
            return cluster_key, ""
        entity_key, attribute_key = cluster_key.rsplit(":", 1)
        return entity_key, attribute_key
