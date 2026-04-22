"""Tests for the MNEMOS guided installer."""

import pytest
import tempfile
from pathlib import Path
from installer.profiles import PROFILES, get_profile, list_profiles, ProfileSpec
from installer.questions import UserAnswers, from_dict
from installer.probes import ProbeResults
from installer.recommend import recommend, Recommendation
from installer.render import render_env, render_manifest


# ─────────────────── Profile Tests ───────────────────


class TestProfiles:
    def test_three_profiles_exist(self):
        assert len(PROFILES) == 3

    def test_profile_names(self):
        assert "core_memory_appliance" in PROFILES
        assert "governance_native" in PROFILES
        assert "custom_manual" in PROFILES

    def test_get_profile(self):
        p = get_profile("core_memory_appliance")
        assert p is not None
        assert p.display_name == "Core Memory Appliance"
        assert p.containers == 3

    def test_get_profile_unknown(self):
        assert get_profile("nonexistent") is None

    def test_list_profiles(self):
        profiles = list_profiles()
        assert len(profiles) == 3
        assert all(isinstance(p, ProfileSpec) for p in profiles)

    def test_core_profile_spec(self):
        p = PROFILES["core_memory_appliance"]
        assert "qdrant" in p.tiers
        assert p.gpu_required is True
        assert p.containers == 3
        assert "qdrant" in p.services

    def test_governance_profile_spec(self):
        p = PROFILES["governance_native"]
        assert "pgvector" in p.tiers
        assert p.containers == 2
        assert "qdrant" not in p.services

    def test_custom_profile_spec(self):
        p = PROFILES["custom_manual"]
        assert p.containers == 0
        assert p.gpu_required is False


# ─────────────────── Questions Tests ───────────────────


class TestQuestions:
    def test_from_dict_defaults(self):
        answers = from_dict({})
        assert answers.use_case == "other"
        assert answers.priority == "semantic_speed"
        assert answers.strict_filters is False
        assert answers.prefer_manual is False

    def test_from_dict_governance(self):
        answers = from_dict({
            "use_case": "compliance_governed",
            "priority": "metadata_governance",
            "strict_filters": "yes",
        })
        assert answers.use_case == "compliance_governed"
        assert answers.priority == "metadata_governance"
        assert answers.strict_filters is True

    def test_from_dict_manual(self):
        answers = from_dict({"prefer_manual": "manual"})
        assert answers.prefer_manual is True


# ─────────────────── Recommendation Tests ───────────────────


def _default_probes(**overrides) -> ProbeResults:
    """Create ProbeResults with sensible defaults."""
    defaults = dict(
        gpu_available=True,
        gpu_name="NVIDIA RTX 4090",
        vram_mb=24576,
        ram_gb=32.0,
        disk_free_gb=100.0,
        docker_available=True,
        nvidia_runtime=True,
        existing_postgres=False,
        os_name="nt",
        cpu_cores=16,
    )
    defaults.update(overrides)
    return ProbeResults(**defaults)


class TestRecommend:
    def test_manual_preference(self):
        """Manual preference → custom_manual."""
        answers = UserAnswers(prefer_manual=True)
        rec = recommend(answers, _default_probes())
        assert rec.profile.name == "custom_manual"
        assert rec.confidence == "high"

    def test_strict_filters_governance(self):
        """Strict filters → governance_native."""
        answers = UserAnswers(strict_filters=True)
        rec = recommend(answers, _default_probes())
        assert rec.profile.name == "governance_native"

    def test_metadata_priority_governance(self):
        """Metadata governance priority → governance_native."""
        answers = UserAnswers(priority="metadata_governance")
        rec = recommend(answers, _default_probes())
        assert rec.profile.name == "governance_native"

    def test_compliance_use_case_governance(self):
        """Compliance use case → governance_native."""
        answers = UserAnswers(use_case="compliance_governed")
        rec = recommend(answers, _default_probes())
        assert rec.profile.name == "governance_native"

    def test_default_core(self):
        """Default (no governance signals) → core_memory_appliance."""
        answers = UserAnswers()
        rec = recommend(answers, _default_probes())
        assert rec.profile.name == "core_memory_appliance"

    def test_agent_memory_core(self):
        """Agent memory → core_memory_appliance."""
        answers = UserAnswers(use_case="agent_memory")
        rec = recommend(answers, _default_probes())
        assert rec.profile.name == "core_memory_appliance"

    def test_no_gpu_warning(self):
        """No GPU → warning but still recommends."""
        answers = UserAnswers()
        rec = recommend(answers, _default_probes(gpu_available=False))
        assert rec.profile.name == "core_memory_appliance"
        assert any("GPU" in w for w in rec.warnings)

    def test_no_docker_warning(self):
        """No Docker → warning."""
        answers = UserAnswers()
        rec = recommend(answers, _default_probes(docker_available=False))
        assert any("Docker" in w for w in rec.warnings)

    def test_no_nvidia_runtime_warning(self):
        """Docker but no NVIDIA runtime → warning."""
        answers = UserAnswers()
        rec = recommend(answers, _default_probes(nvidia_runtime=False))
        assert any("NVIDIA" in w for w in rec.warnings)

    def test_low_ram_warning(self):
        """Low RAM → warning."""
        answers = UserAnswers()
        rec = recommend(answers, _default_probes(ram_gb=0.5))
        assert any("RAM" in w for w in rec.warnings)

    def test_low_disk_warning(self):
        """Low disk → warning."""
        answers = UserAnswers()
        rec = recommend(answers, _default_probes(disk_free_gb=2.0))
        assert any("disk" in w.lower() for w in rec.warnings)

    def test_confidence_degrades_with_warnings(self):
        """Multiple warnings → medium confidence."""
        answers = UserAnswers()
        rec = recommend(answers, _default_probes(
            gpu_available=False, docker_available=False,
        ))
        assert rec.confidence in ("medium", "low")

    def test_alternatives_populated(self):
        """Recommendation includes alternatives."""
        answers = UserAnswers()
        rec = recommend(answers, _default_probes())
        assert len(rec.alternatives) > 0

    def test_governance_reasons_populated(self):
        """Governance recommendation has reasons."""
        answers = UserAnswers(strict_filters=True, use_case="compliance_governed")
        rec = recommend(answers, _default_probes())
        assert len(rec.reasons) >= 2
        assert any("pgvector" in r.lower() for r in rec.reasons)


class TestRenderHybridSettings:
    def test_render_env_includes_hybrid_controls(self):
        profile = get_profile("core_memory_appliance")
        assert profile is not None

        with tempfile.TemporaryDirectory(dir=".") as tmpdir:
            env_path = render_env(
                profile,
                Path(tmpdir),
                retrieval_mode="hybrid",
                fusion_policy="lexical_dominant",
                lexical_top_k=40,
                semantic_top_k=30,
                explain_default=True,
            )

            content = env_path.read_text()
            assert "MNEMOS_RETRIEVAL_MODE=hybrid" in content
            assert "MNEMOS_FUSION_POLICY=lexical_dominant" in content
            assert "MNEMOS_LEXICAL_TOP_K=40" in content
            assert "MNEMOS_SEMANTIC_TOP_K=30" in content
            assert "MNEMOS_EXPLAIN_DEFAULT=true" in content

    def test_render_manifest_captures_retrieval_settings(self):
        rec = Recommendation(
            profile=PROFILES["governance_native"],
            confidence="high",
            reasons=["test"],
        )
        answers = UserAnswers()
        probes = _default_probes()

        with tempfile.TemporaryDirectory(dir=".") as tmpdir:
            manifest_path = render_manifest(
                rec,
                answers,
                probes,
                Path(tmpdir),
                retrieval_mode="hybrid",
                fusion_policy="balanced",
                lexical_top_k=25,
                semantic_top_k=25,
                explain_default=False,
            )

            content = manifest_path.read_text()
            assert "retrieval:" in content
            assert "mode: hybrid" in content
            assert "fusion_policy: balanced" in content
