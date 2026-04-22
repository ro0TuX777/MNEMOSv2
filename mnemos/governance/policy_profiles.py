"""
Tenant-aware governance policy profiles.

Profiles allow per-tenant tuning of:
- read-path thresholds (score floor, freshness half-life)
- reflect-path precision (overlap threshold + token guards)
- reinforcement deltas (utility/trust/stability updates)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from mnemos.governance.reinforcement import ReinforcementConfig


@dataclass(frozen=True)
class GovernancePolicyProfile:
    profile_id: str
    min_score_threshold: float = 0.0
    freshness_half_life_days: float = 180.0
    overlap_threshold: float = 0.15
    min_memory_tokens_for_overlap: int = 3
    min_overlap_tokens: int = 2
    utility_used: float = 0.05
    utility_ignored: float = -0.01
    utility_contradiction_loser: float = -0.03
    trust_used: float = 0.02
    trust_ignored: float = 0.0
    trust_contradiction_loser: float = -0.02
    stability_used: float = 0.02

    @classmethod
    def from_mapping(
        cls,
        profile_id: str,
        data: Mapping[str, object],
        *,
        base: Optional["GovernancePolicyProfile"] = None,
    ) -> "GovernancePolicyProfile":
        seed = base or GovernancePolicyProfile(profile_id=profile_id)
        return cls(
            profile_id=profile_id,
            min_score_threshold=float(data.get("min_score_threshold", seed.min_score_threshold)),
            freshness_half_life_days=float(data.get("freshness_half_life_days", seed.freshness_half_life_days)),
            overlap_threshold=float(data.get("overlap_threshold", seed.overlap_threshold)),
            min_memory_tokens_for_overlap=int(
                data.get("min_memory_tokens_for_overlap", seed.min_memory_tokens_for_overlap)
            ),
            min_overlap_tokens=int(data.get("min_overlap_tokens", seed.min_overlap_tokens)),
            utility_used=float(data.get("utility_used", seed.utility_used)),
            utility_ignored=float(data.get("utility_ignored", seed.utility_ignored)),
            utility_contradiction_loser=float(
                data.get("utility_contradiction_loser", seed.utility_contradiction_loser)
            ),
            trust_used=float(data.get("trust_used", seed.trust_used)),
            trust_ignored=float(data.get("trust_ignored", seed.trust_ignored)),
            trust_contradiction_loser=float(
                data.get("trust_contradiction_loser", seed.trust_contradiction_loser)
            ),
            stability_used=float(data.get("stability_used", seed.stability_used)),
        ).validate()

    def validate(self) -> "GovernancePolicyProfile":
        if self.freshness_half_life_days < 1.0:
            raise ValueError("freshness_half_life_days must be >= 1.0")
        if self.min_score_threshold < 0.0:
            raise ValueError("min_score_threshold must be >= 0.0")
        if not (0.0 <= self.overlap_threshold <= 1.0):
            raise ValueError("overlap_threshold must be in [0.0, 1.0]")
        if self.min_memory_tokens_for_overlap < 1:
            raise ValueError("min_memory_tokens_for_overlap must be >= 1")
        if self.min_overlap_tokens < 1:
            raise ValueError("min_overlap_tokens must be >= 1")
        return self

    def reinforcement_config(self) -> ReinforcementConfig:
        return ReinforcementConfig(
            utility_used=self.utility_used,
            utility_ignored=self.utility_ignored,
            utility_contradiction_loser=self.utility_contradiction_loser,
            trust_used=self.trust_used,
            trust_ignored=self.trust_ignored,
            trust_contradiction_loser=self.trust_contradiction_loser,
            stability_used=self.stability_used,
        )


def load_policy_profiles(
    *,
    raw_json: str,
    base_min_score_threshold: float,
    base_freshness_half_life_days: float,
) -> Dict[str, GovernancePolicyProfile]:
    """
    Parse tenant policy profiles from JSON.

    Expected shape:
    {
      "default": {...optional overrides...},
      "tenant_a": {...},
      "tenant_b": {...}
    }
    """
    default_profile = GovernancePolicyProfile(
        profile_id="default",
        min_score_threshold=base_min_score_threshold,
        freshness_half_life_days=base_freshness_half_life_days,
    ).validate()
    profiles: Dict[str, GovernancePolicyProfile] = {"default": default_profile}

    text = (raw_json or "").strip()
    if not text:
        return profiles

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("MNEMOS_GOVERNANCE_POLICY_PROFILES_JSON must be a JSON object")

    if "default" in parsed:
        default_raw = parsed.get("default", {})
        if not isinstance(default_raw, dict):
            raise ValueError("default governance profile must be an object")
        profiles["default"] = GovernancePolicyProfile.from_mapping(
            "default",
            default_raw,
            base=default_profile,
        )

    for key, value in parsed.items():
        if key == "default":
            continue
        if not isinstance(key, str) or not key.strip():
            raise ValueError("governance profile names must be non-empty strings")
        if not isinstance(value, dict):
            raise ValueError(f"governance profile '{key}' must be an object")
        profiles[key] = GovernancePolicyProfile.from_mapping(
            key,
            value,
            base=profiles["default"],
        )

    return profiles
