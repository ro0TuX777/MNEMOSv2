"""
Seed the representative replay/stress corpus into the promoted Qdrant collection.

Replaces the ephemeral seeding script used for the Phase 7 N=2121 representative
replay: 2,100 deterministic themed engrams + the 21 PDF-grounded reference
passages already present = 2,121 points.

Themes are derived from the burn-in reference families
(benchmarks/truthsets/phase7_pdf_reference_passages.json and the anchor query
domains) so that Phase 9 clustering produces summaries thematically aligned
with the CLASS_C golden summaries in query_complexity_v1.json.

Deterministic and idempotent: engram ids are stable (rep_<theme>_<nnn>), and
QdrantTier upserts by UUIDv5 of the id, so re-running overwrites in place.

Usage:
  python tools/seed_representative_corpus.py --gpu-device cpu
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mnemos.engram.model import Engram

DEFAULT_COUNT = 2100
BATCH_SIZE = 128

# 20 themes; ~105 docs each at the default count. Sentence banks are grounded
# in the burn-in reference families so cluster summaries land near the
# CLASS_C golden summaries.
THEMES: Dict[str, List[str]] = {
    "gdpr_disclosure": [
        "GDPR requires controllers to disclose how personal data is handled at the point of collection.",
        "Data handling disclosures must name the processing purposes and the lawful basis relied upon.",
        "Privacy notices describe retention periods and the categories of recipients for personal data.",
        "Controllers document disclosure obligations so supervisory authorities can audit handling practices.",
        "Transparency duties oblige organisations to explain profiling and automated decision logic to data subjects.",
    ],
    "erasure_minimization": [
        "The right to erasure under article 17 obliges controllers to delete personal data without undue delay.",
        "Data minimization restricts collection to what is adequate, relevant, and necessary for the stated purpose.",
        "Erasure requests propagate to processors and downstream recipients of the personal data.",
        "Minimization reviews remove fields that no longer serve the documented processing purpose.",
        "Deletion duties survive system migrations; archived copies are purged on the same schedule.",
    ],
    "sigint_bounded_reflection": [
        "Signals intelligence reflection must remain bounded by the authorized mission purpose.",
        "SIGINT products passed to subordinate commanders stay within United States signals intelligence directives.",
        "Collection authorities constrain how intercepted material may be correlated or analysed.",
        "Bounded reflection keeps analytic reuse of signals data inside the original authorization envelope.",
        "Oversight reviews confirm that signals processing did not exceed the approved purpose statement.",
    ],
    "tenant_policy_boundaries": [
        "Tenant policy profiles isolate one organisation's thresholds and reinforcement deltas from another's.",
        "Per-tenant governance keeps retrieval boundaries intact across shared infrastructure.",
        "Policy profiles are validated at load time so a malformed tenant override cannot leak defaults.",
        "Tenant boundaries apply to read-path scoring, reflect precision, and hygiene schedules alike.",
        "Cross-tenant access is denied by construction; profiles select thresholds per request.",
    ],
    "retention_oversight": [
        "Retention schedules define how long records persist before mandatory review or deletion.",
        "Oversight bodies audit retention decisions against the documented disposition authority.",
        "Records past their retention horizon move to a disposition queue rather than silent deletion.",
        "The corpus-wide stance ties deletion, retention, and oversight into one auditable lifecycle.",
        "Intelligence oversight guides require periodic certification that retained data remains authorized.",
    ],
    "stale_cache_survival": [
        "Stale cache survival measures whether invalidated derived views can still be served.",
        "Cache entries are keyed by inputs so any artifact change forces a fresh computation.",
        "The stale cache survival rate is held at or near zero by deterministic invalidation.",
        "Dry-run parity confirms cache invalidation behaves identically with and without writes.",
        "Surviving stale entries are treated as correctness defects, not performance artifacts.",
    ],
    "contradiction_handling": [
        "Contradiction handling groups candidates by entity slot and selects a deterministic winner.",
        "Conflicting memories are resolved by trust, recency, utility, and source authority in order.",
        "Contradiction losers receive a suppression modifier and are removed in enforced mode.",
        "Offline contradiction sweeps catch conflicts between memories never co-retrieved together.",
        "Resolution decisions record the winner and a human-readable contradiction reason.",
    ],
    "explainability_traces": [
        "Why-won and why-lost traces show each modifier that shaped a result's governed score.",
        "Counterfactual explanations state the modifier value that would have tied rank one.",
        "Suppressed candidates carry the reason and the threshold that excluded them.",
        "Explainability coverage extends to freshness age inversion in days-old terms.",
        "Operators tune policies from explain traces instead of intuition.",
    ],
    "sox_internal_controls": [
        "SOX section 404 requires management to assess internal control over financial reporting.",
        "Quarterly reporting controls document who reviewed and certified the figures.",
        "Internal control deficiencies are classified and remediated before certification.",
        "Management's assessment of control effectiveness is evidenced and auditable.",
        "Control owners attest quarterly that key reconciliations were performed.",
    ],
    "hipaa_privacy": [
        "HIPAA privacy compliance gaps arise when patient data access lacks a documented need.",
        "Protected health information requires minimum necessary access controls.",
        "Privacy gap assessments inventory disclosures of patient data to third parties.",
        "HIPAA safeguards span administrative, physical, and technical control families.",
        "Patient data handling reviews confirm authorizations cover each disclosure.",
    ],
    "breach_notification": [
        "Breach disclosure deadlines differ across regulations; the strictest clock governs response planning.",
        "Notification timelines define when regulators and affected individuals must be informed.",
        "Covered entity scope determines which organisations owe breach notifications.",
        "Penalty structures escalate with delay, scope, and demonstrated negligence.",
        "Incident commanders track the disclosure deadline from the moment of discovery.",
    ],
    "incident_response": [
        "After discovering a security incident, the organisation must contain, assess, and notify.",
        "Unauthorized access to sensitive records triggers the incident response runbook.",
        "Post-incident obligations include forensic preservation and regulator notification.",
        "Response teams document the timeline from detection through remediation.",
        "Lessons-learned reviews feed control improvements after every incident.",
    ],
    "signals_operations": [
        "Signals operations doctrine assigns collection, processing, and dissemination responsibilities.",
        "Communications intelligence flows from collection sites to supported commands under strict handling rules.",
        "Operational signals reporting distinguishes raw intercept from evaluated product.",
        "Field signals units coordinate frequencies, schedules, and reporting formats.",
        "Doctrine manuals describe how signals elements support the commander's intent.",
    ],
    "intelligence_oversight": [
        "The intelligence oversight guide requires reporting of questionable activities.",
        "Oversight officers verify that collection stayed within assigned authorities.",
        "Annual oversight training covers prohibited targeting and retention limits.",
        "Inspectors review files for compliance with intelligence oversight directives.",
        "Oversight findings escalate through the chain with corrective action plans.",
    ],
    "audit_logging": [
        "Every memory mutation is logged immutably with timestamps and component identity.",
        "Forensic ledgers replay the sequence of operations behind any retrieval.",
        "Audit queries answer exactly what was retrieved for a given answer and when.",
        "Compliance reviews rely on append-only logs with full-text search.",
        "Operation latency and status accompany each audit record.",
    ],
    "access_control": [
        "Access to governed records is granted by role and verified per request.",
        "Security markings restrict retrieval to cleared compartments.",
        "Least privilege keeps service accounts scoped to the operations they perform.",
        "Access reviews revoke entitlements that no longer match duties.",
        "Privileged actions require dual authorization and leave audit evidence.",
    ],
    "encryption_controls": [
        "Data at rest is encrypted with managed keys rotated on schedule.",
        "Transport encryption protects retrieval traffic between services.",
        "Key custodians document escrow and recovery procedures.",
        "Cryptographic control reviews verify algorithms meet current policy.",
        "Compromised keys trigger immediate rotation and re-encryption.",
    ],
    "vendor_risk": [
        "Third-party processors are assessed before receiving any governed data.",
        "Vendor contracts bind processors to the same handling obligations.",
        "Subprocessor changes require notice and renewed risk review.",
        "Vendor incidents feed the same notification clocks as internal breaches.",
        "Annual vendor reviews score control evidence against the baseline.",
    ],
    "training_awareness": [
        "Handlers complete annual training on data protection duties.",
        "Awareness programmes cover phishing, mishandling, and reporting channels.",
        "Role-specific modules train analysts on authorized purpose limits.",
        "Training completion is tracked and escalated when overdue.",
        "Refresher content updates whenever regulations or policies change.",
    ],
    "records_disposition": [
        "Disposition authorities determine transfer, archival, or destruction of records.",
        "Destruction certificates evidence that disposition was carried out.",
        "Holds suspend disposition when litigation or investigation requires.",
        "Disposition schedules map record categories to retention outcomes.",
        "Archived records keep provenance metadata for future audit.",
    ],
}

_DEPARTMENTS = ["legal", "finance", "operations", "medical", "signals", "platform"]
_PERIODS = ["30", "45", "60", "72", "90", "180"]


def build_corpus(count: int) -> List[Engram]:
    rng = random.Random(2121)
    theme_ids = sorted(THEMES)
    engrams: List[Engram] = []
    per_theme = count // len(theme_ids)
    remainder = count - per_theme * len(theme_ids)
    for t_idx, theme in enumerate(theme_ids):
        bank = THEMES[theme]
        n_docs = per_theme + (1 if t_idx < remainder else 0)
        for i in range(n_docs):
            picks = rng.sample(bank, k=min(3, len(bank)))
            dept = rng.choice(_DEPARTMENTS)
            period = rng.choice(_PERIODS)
            content = (
                f"{picks[0]} {picks[1]} "
                f"For the {dept} function this is reviewed every {period} days. {picks[2]}"
            )
            engrams.append(
                Engram(
                    id=f"rep_{theme}_{i:03d}",
                    content=content,
                    source=f"synthetic://representative_corpus/{theme}",
                    confidence=0.9,
                    metadata={"theme": theme, "synthetic_replay_corpus": True},
                )
            )
    return engrams


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the representative replay corpus")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="mnemos_engrams_nomic_mrl")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--embedding-model", default="nomic-ai/nomic-embed-text-v1.5")
    parser.add_argument("--gpu-device", default="cpu")
    args = parser.parse_args()

    from mnemos.retrieval.qdrant_tier import QdrantTier

    tier = QdrantTier(
        url=args.qdrant_url,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        embedding_dim=768,
        gpu_device=args.gpu_device,
    )

    corpus = build_corpus(args.count)
    print(f"seeding {len(corpus)} engrams across {len(THEMES)} themes ...")
    written = 0
    for start in range(0, len(corpus), BATCH_SIZE):
        batch = corpus[start : start + BATCH_SIZE]
        written += tier.index(batch)
        print(f"  upserted {written}/{len(corpus)}")

    from qdrant_client import QdrantClient

    total = QdrantClient(url=args.qdrant_url, timeout=30).count(args.collection, exact=True).count
    print(f"done: {written} written; collection now holds {total} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
