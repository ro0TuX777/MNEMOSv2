"""
MNEMOS Benchmark - Corpus Generator
=====================================

Generates structured synthetic corpora with realistic metadata
for profile retrieval benchmarking.
"""

import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnemos.engram.model import Engram


# ─────────────── Domain content templates ───────────────

DOMAINS = {
    "finance": {
        "templates": [
            "The quarterly revenue exceeded {amount} million, driven by {driver}.",
            "Risk assessment for {entity} indicates {risk_level} exposure in {area}.",
            "Capital allocation policy requires {percent}% reserves for {category}.",
            "Audit findings on {entity} revealed {finding} in fiscal year {year}.",
            "The portfolio rebalancing strategy targets {target} across {asset_class}.",
            "Compliance review of {entity} identified {count} material discrepancies.",
            "Budget forecast for {year} projects {direction} growth in {segment}.",
            "The interest rate sensitivity analysis shows {basis_points}bp impact on {metric}.",
            "Tax provision for {entity} requires adjustment of {amount} million.",
            "Internal controls testing found {status} effectiveness in {area}.",
        ],
        "tags": ["finance", "quarterly", "compliance", "audit", "budget", "risk", "capital"],
        "vars": {
            "amount": ["12.4", "58.7", "124.3", "3.2", "890.1"],
            "driver": ["cloud services", "retail expansion", "cost optimization", "new markets"],
            "entity": ["Division A", "Subsidiary B", "Fund C", "Portfolio D"],
            "risk_level": ["moderate", "high", "low", "elevated"],
            "area": ["credit risk", "market risk", "operational risk", "liquidity"],
            "percent": ["10", "15", "20", "25"],
            "category": ["loan losses", "market volatility", "operational failures"],
            "finding": ["three overstatements", "two misclassifications", "one omission"],
            "year": ["2024", "2025", "2026"],
            "target": ["60/40 equity-bond", "risk-parity", "factor-weighted"],
            "asset_class": ["fixed income", "equities", "alternatives"],
            "count": ["3", "7", "12"],
            "direction": ["moderate", "strong", "flat"],
            "segment": ["enterprise", "consumer", "institutional"],
            "basis_points": ["25", "50", "75", "100"],
            "metric": ["NII", "EVE", "capital ratio"],
            "status": ["adequate", "insufficient", "strong"],
        },
    },
    "legal": {
        "templates": [
            "Contract {ref} Section {section} governs {clause_type} obligations for {party}.",
            "The regulatory filing for {jurisdiction} requires disclosure of {disclosure}.",
            "Litigation reserve for {case} is estimated at {amount} million.",
            "Intellectual property review of {asset} confirms {status} protection.",
            "Employment agreement with {role} includes {provision} provisions.",
            "Data privacy assessment under {regulation} identified {count} gaps.",
            "Merger due diligence for {entity} requires review of {doc_count} documents.",
            "The compliance training module covers {topic} requirements for {audience}.",
            "Board resolution {ref} authorises {action} effective {date}.",
            "Subpoena response for {case} involves {page_count} pages of discovery.",
        ],
        "tags": ["legal", "contract", "compliance", "regulatory", "litigation", "privacy"],
        "vars": {
            "ref": ["MN-2024-001", "MN-2024-042", "MN-2025-017", "MN-2025-088"],
            "section": ["4.2", "7.1", "12.3", "3.6"],
            "clause_type": ["indemnification", "termination", "confidentiality", "liability"],
            "party": ["the Licensee", "the Vendor", "the Acquirer"],
            "jurisdiction": ["EU", "US-Federal", "UK", "APAC"],
            "disclosure": ["beneficial ownership", "related-party transactions", "risk factors"],
            "case": ["Smith v. Corp", "Doe v. Platform", "SEC Investigation"],
            "amount": ["2.4", "15.8", "42.0"],
            "asset": ["patent portfolio", "trademark family", "trade secrets"],
            "status": ["adequate", "partial", "expired"],
            "role": ["CTO", "CFO", "General Counsel"],
            "provision": ["non-compete", "severance", "change of control"],
            "regulation": ["GDPR", "CCPA", "HIPAA", "SOX"],
            "count": ["4", "8", "14"],
            "entity": ["TargetCo", "AcquireCo", "JointVenture LLC"],
            "doc_count": ["1,200", "4,500", "12,000"],
            "topic": ["anti-bribery", "insider trading", "data handling"],
            "audience": ["all employees", "senior management", "board members"],
            "action": ["share buyback", "dividend declaration", "executive appointment"],
            "date": ["2025-01-15", "2025-06-01", "2026-03-01"],
            "page_count": ["3,400", "12,000", "850"],
        },
    },
    "medical": {
        "templates": [
            "Clinical trial {trial_id} for {compound} shows {outcome} in {phase}.",
            "Patient cohort analysis indicates {finding} in the {population} group.",
            "Adverse event report for {drug} documents {event_type} in {count} subjects.",
            "The diagnostic accuracy of {test} is {accuracy}% for {condition}.",
            "Treatment protocol for {condition} was updated based on {evidence_type} evidence.",
            "Genomic analysis of {sample} identified {variant_count} actionable variants.",
            "Hospital readmission rate for {department} decreased by {percent}%.",
            "Drug interaction warning: {drug_a} with {drug_b} may cause {effect}.",
            "The imaging review of {patient_group} revealed {finding} in {percent}% of cases.",
            "Post-market surveillance of {device} reports {incident_count} incidents.",
        ],
        "tags": ["medical", "clinical", "trial", "patient", "diagnostic", "treatment"],
        "vars": {
            "trial_id": ["NCT-0042", "NCT-0187", "NCT-0513"],
            "compound": ["MN-201", "MN-305", "AB-112"],
            "outcome": ["statistically significant improvement", "non-inferiority", "no benefit"],
            "phase": ["Phase II", "Phase III", "Phase IV"],
            "finding": ["elevated biomarkers", "improved outcomes", "no significant difference"],
            "population": ["pediatric", "geriatric", "immunocompromised"],
            "drug": ["Compound X", "Drug Y", "Therapy Z"],
            "event_type": ["hepatotoxicity", "cardiac events", "dermatologic reactions"],
            "count": ["3", "12", "27"],
            "test": ["CT angiography", "MRI spectroscopy", "blood panel"],
            "accuracy": ["94.2", "87.5", "91.8"],
            "condition": ["Type 2 diabetes", "early-stage melanoma", "chronic heart failure"],
            "evidence_type": ["randomized controlled", "meta-analysis", "real-world"],
            "sample": ["biopsy specimen", "blood draw", "tissue culture"],
            "variant_count": ["3", "7", "12"],
            "department": ["cardiology", "oncology", "emergency"],
            "percent": ["14", "22", "8"],
            "drug_a": ["warfarin", "metformin", "lisinopril"],
            "drug_b": ["aspirin", "fluconazole", "amiodarone"],
            "effect": ["bleeding risk", "hypoglycemia", "QT prolongation"],
            "patient_group": ["chest pain cohort", "stroke risk group", "post-surgical"],
            "device": ["cardiac stent model A", "insulin pump B", "defibrillator C"],
            "incident_count": ["4", "11", "2"],
        },
    },
    "technical": {
        "templates": [
            "Service {service} experienced {issue} with {metric} degradation of {percent}%.",
            "The deployment pipeline for {project} uses {tool} with {strategy} strategy.",
            "Memory profiling of {component} shows {leak_size} MB leak per {interval}.",
            "API endpoint {endpoint} has p99 latency of {latency}ms under {load}.",
            "Database migration {version} adds {table_count} tables and {index_count} indexes.",
            "The caching layer for {service} achieves {hit_rate}% hit rate with {eviction} eviction.",
            "Container image for {service} reduced from {old_size} to {new_size} MB.",
            "Load test results: {service} handles {rps} req/s at {cpu}% CPU utilisation.",
            "Security scan of {component} found {vuln_count} vulnerabilities ({severity}).",
            "The monitoring stack uses {tool} for {purpose} with {retention} retention.",
        ],
        "tags": ["technical", "infrastructure", "deployment", "performance", "security"],
        "vars": {
            "service": ["auth-service", "data-pipeline", "api-gateway", "ml-inference"],
            "issue": ["connection pool exhaustion", "OOM kill", "timeout spike"],
            "metric": ["throughput", "latency", "error rate"],
            "percent": ["12", "35", "8", "62"],
            "project": ["platform-v3", "data-lake", "ml-ops"],
            "tool": ["ArgoCD", "Jenkins", "GitHub Actions"],
            "strategy": ["blue-green", "canary", "rolling"],
            "component": ["embedding-engine", "query-router", "cache-layer"],
            "leak_size": ["24", "128", "512"],
            "interval": ["hour", "day", "request cycle"],
            "endpoint": ["/v1/search", "/v1/ingest", "/healthz"],
            "latency": ["42", "185", "12"],
            "load": ["1000 req/s", "5000 req/s", "200 req/s"],
            "version": ["v2.4", "v3.0", "v1.12"],
            "table_count": ["3", "7", "1"],
            "index_count": ["5", "12", "2"],
            "hit_rate": ["94", "87", "99"],
            "eviction": ["LRU", "LFU", "TTL"],
            "old_size": ["1200", "850", "2400"],
            "new_size": ["340", "210", "680"],
            "rps": ["2400", "8500", "12000"],
            "cpu": ["45", "72", "88"],
            "vuln_count": ["2", "7", "0"],
            "severity": ["1 critical, 1 high", "3 medium, 4 low", "none"],
            "purpose": ["metrics", "tracing", "log aggregation"],
            "retention": ["30 day", "90 day", "1 year"],
        },
    },
}

DEPARTMENTS = ["finance", "legal", "engineering", "medical"]
TENANTS = ["tenant-A", "tenant-B", "tenant-C", "tenant-D", "tenant-E"]
CLEARANCES = ["public", "internal", "restricted", "classified"]


def _fill_template(template: str, variables: dict, rng: random.Random) -> str:
    """Fill a template string with random variable selections."""
    result = template
    for key, values in variables.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, rng.choice(values), 1)
    return result


def generate_corpus(
    n_docs: int = 10_000,
    seed: int = 42,
    domains: Optional[List[str]] = None,
) -> List[Engram]:
    """
    Generate a synthetic structured corpus of engrams.

    Each engram has realistic content and structured metadata
    suitable for filtered retrieval benchmarks.
    """
    rng = random.Random(seed)
    if domains is None:
        domains = list(DOMAINS.keys())

    engrams = []
    for i in range(n_docs):
        domain = rng.choice(domains)
        domain_data = DOMAINS[domain]

        template = rng.choice(domain_data["templates"])
        content = _fill_template(template, domain_data["vars"], rng)

        # Deterministic ID from content + index
        doc_id = hashlib.sha256(f"{seed}:{i}:{content}".encode()).hexdigest()[:16]

        # Structured metadata
        department = domain if domain != "technical" else rng.choice(["engineering", "finance"])
        tenant = rng.choice(TENANTS)
        clearance = rng.choice(CLEARANCES)

        engram = Engram(
            id=doc_id,
            content=content,
            source=f"dept-{department}",
            neuro_tags=rng.sample(domain_data["tags"], k=min(rng.randint(2, 4), len(domain_data["tags"]))),
            confidence=round(rng.uniform(0.5, 1.0), 2),
            metadata={
                "department": department,
                "tenant": tenant,
                "clearance": clearance,
                "domain": domain,
            },
        )
        engrams.append(engram)

    return engrams


def save_corpus(engrams: List[Engram], path: Path):
    """Save corpus to JSON for reproducibility."""
    data = []
    for e in engrams:
        data.append({
            "id": e.id,
            "content": e.content,
            "source": e.source,
            "neuro_tags": e.neuro_tags,
            "confidence": e.confidence,
            "metadata": e.metadata,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_corpus(path: Path) -> List[Engram]:
    """Load corpus from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [
        Engram(
            id=d["id"],
            content=d["content"],
            source=d["source"],
            neuro_tags=d.get("neuro_tags", []),
            confidence=d.get("confidence", 0.8),
            metadata=d.get("metadata", {}),
        )
        for d in data
    ]
