from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.verify_mnemos_local_stack import (
    CommandResult,
    HttpResult,
    StackConfig,
    run_preflight,
    write_receipt,
)


class FakeRunner:
    def __init__(self, *, postgres_ok: bool = True) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.postgres_ok = postgres_ok
        self.config = {
            "services": {
                "qdrant": {"image": "qdrant/qdrant:v1.17.1"},
                "postgres": {
                    "image": "postgres:16-alpine",
                    "environment": {
                        "POSTGRES_USER": "mnemos",
                        "POSTGRES_DB": "mnemos",
                        "POSTGRES_PASSWORD": "do-not-leak",
                    },
                },
                "mnemos": {"image": "local/mnemos:test"},
                "research-ui": {"image": "local/mnemos:test"},
                "openwebui-proxy": {"image": "local/mnemos:test"},
            }
        }
        self.rows = [
            self._row("qdrant", "random-qdrant-1", "qdrant/qdrant:v1.17.1", 6333, 6333),
            self._row("postgres", "random-postgres-1", "postgres:16-alpine", 5432, 5432),
            self._row("mnemos", "random-project-mnemos-1", "local/mnemos:test", 8700, 8700),
            self._row("research-ui", "random-research-1", "local/mnemos:test", 8788, 8788),
            self._row("openwebui-proxy", "random-proxy-1", "local/mnemos:test", 8790, 8790),
        ]

    @staticmethod
    def _row(service: str, name: str, image: str, target: int, published: int) -> dict:
        return {
            "Service": service,
            "Name": name,
            "Image": image,
            "State": "running",
            "Publishers": [{"TargetPort": target, "PublishedPort": published, "Protocol": "tcp"}],
            "Labels": f"com.docker.compose.service={service}",
        }

    def run(self, args: list[str], timeout_s: float) -> CommandResult:
        call = tuple(args)
        self.calls.append(call)
        if "config" in args:
            return CommandResult(0, json.dumps(self.config), "")
        if "ps" in args:
            return CommandResult(0, json.dumps(self.rows), "")
        if "pg_isready" in args:
            if self.postgres_ok:
                return CommandResult(0, "/var/run/postgresql:5432 - accepting connections", "")
            return CommandResult(1, "", "no response")
        raise AssertionError(f"unexpected command: {args}")


class FakeHttp:
    def __init__(self) -> None:
        self.failures: dict[str, str] = {}
        self.responses = {
            "http://127.0.0.1:6333/healthz": "healthz check passed",
            "http://127.0.0.1:8700/health": json.dumps({"status": "healthy"}),
            "http://127.0.0.1:8700/v1/mnemos/capabilities": json.dumps({"status": "healthy"}),
            "http://127.0.0.1:8788/": "MNEMOS Research UI",
            "http://127.0.0.1:8790/health": json.dumps({"status": "healthy"}),
        }

    def get(self, url: str, timeout_s: float) -> HttpResult:
        if url in self.failures:
            return HttpResult(False, None, "", 1.5, self.failures[url])
        return HttpResult(True, 200, self.responses[url], 1.0, None)


def config(**changes) -> StackConfig:
    values = {"compose_file": Path("docker-compose.yml")}
    values.update(changes)
    return StackConfig(**values)


def test_compose_roles_are_discovered_without_fixed_container_names():
    receipt = run_preflight(config(), FakeRunner(), FakeHttp())
    assert receipt.ok is True
    assert receipt.services["mnemos"].container_name == "random-project-mnemos-1"
    assert receipt.services["mnemos"].compose_service == "mnemos"


def test_required_connection_failure_is_actionable_and_nonzero():
    http = FakeHttp()
    http.failures["http://127.0.0.1:8700/health"] = "connection refused"
    receipt = run_preflight(config(), FakeRunner(), http)
    check = receipt.services["mnemos"]
    assert receipt.ok is False
    assert check.reason_code == "HTTP_CONNECTION_FAILED"
    assert "8700" in check.remediation


def test_optional_absent_differs_from_configured_unhealthy():
    runner = FakeRunner()
    del runner.config["services"]["research-ui"]
    del runner.config["services"]["openwebui-proxy"]
    runner.rows = [row for row in runner.rows if row["Service"] not in {"research-ui", "openwebui-proxy"}]
    receipt = run_preflight(config(research_ui_url=None, proxy_url=None), runner, FakeHttp())
    assert receipt.services["research-ui"].status == "not_configured"
    assert receipt.services["openwebui-proxy"].status == "not_configured"
    assert receipt.ok is True


def test_verifier_never_runs_mutating_docker_commands():
    runner = FakeRunner()
    run_preflight(config(), runner, FakeHttp())
    commands = [" ".join(call) for call in runner.calls]
    forbidden = (" up ", " down ", " restart ", " rm ", " pull ", " build ")
    assert not any(word in f" {command} " for command in commands for word in forbidden)


def test_qdrant_image_mismatch_fails_closed():
    runner = FakeRunner()
    runner.config["services"]["qdrant"]["image"] = "qdrant/qdrant:v1.16.0"
    receipt = run_preflight(config(), runner, FakeHttp())
    assert receipt.services["qdrant"].reason_code == "IMAGE_MISMATCH"
    assert receipt.ok is False


def test_postgres_readiness_failure_is_reported():
    receipt = run_preflight(config(), FakeRunner(postgres_ok=False), FakeHttp())
    assert receipt.services["postgres"].reason_code == "POSTGRES_NOT_READY"
    assert receipt.ok is False


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (json.dumps({"status": "degraded"}), "SERVICE_DEGRADED"),
        ("not-json", "INVALID_CAPABILITIES_JSON"),
    ],
)
def test_mnemos_capability_failures_are_distinguished(body: str, reason: str):
    http = FakeHttp()
    if reason == "SERVICE_DEGRADED":
        http.responses["http://127.0.0.1:8700/health"] = body
    else:
        http.responses["http://127.0.0.1:8700/v1/mnemos/capabilities"] = body
    receipt = run_preflight(config(), FakeRunner(), http)
    assert receipt.services["mnemos"].reason_code == reason


def test_published_port_ownership_mismatch_is_not_hidden_by_reachable_http():
    runner = FakeRunner()
    mnemos = next(row for row in runner.rows if row["Service"] == "mnemos")
    mnemos["Publishers"][0]["PublishedPort"] = 18700
    receipt = run_preflight(config(), runner, FakeHttp())
    assert receipt.services["mnemos"].reason_code == "PORT_OWNERSHIP_MISMATCH"
    assert receipt.ok is False


def test_custom_compose_file_urls_and_timeout_are_used():
    runner = FakeRunner()
    http = FakeHttp()
    http.responses["http://localhost:9700/health"] = json.dumps({"status": "healthy"})
    http.responses["http://localhost:9700/v1/mnemos/capabilities"] = json.dumps({"status": "healthy"})
    runner.rows[2]["Publishers"][0]["PublishedPort"] = 9700
    receipt = run_preflight(
        config(compose_file=Path("installer.yml"), mnemos_url="http://localhost:9700", timeout_s=3.5),
        runner,
        http,
    )
    assert receipt.ok is True
    assert runner.calls[0][:4] == ("docker", "compose", "-f", "installer.yml")


def test_receipt_redacts_sensitive_environment_and_is_non_host_specific(tmp_path: Path):
    receipt = run_preflight(config(), FakeRunner(), FakeHttp())
    output = tmp_path / "receipt.json"
    write_receipt(receipt, output)
    text = output.read_text(encoding="utf-8")
    assert "do-not-leak" not in text
    assert "POSTGRES_PASSWORD" not in text
    assert receipt.to_dict()["services"]["postgres"]["status"] == "healthy"


@pytest.mark.parametrize(
    ("role", "url", "required"),
    [
        ("research-ui", "http://127.0.0.1:8788/", "require_research_ui"),
        ("openwebui-proxy", "http://127.0.0.1:8790/health", "require_proxy"),
    ],
)
def test_optional_endpoint_failure_becomes_required_failure(role: str, url: str, required: str):
    http = FakeHttp()
    http.failures[url] = "timed out"
    receipt = run_preflight(config(**{required: True}), FakeRunner(), http)
    assert receipt.services[role].reason_code == "HTTP_CONNECTION_FAILED"
    assert receipt.ok is False
