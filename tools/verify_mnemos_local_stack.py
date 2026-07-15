"""Read-only connectivity preflight for a local MNEMOS Docker Compose stack."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class HttpResult:
    ok: bool
    status_code: int | None
    body: str
    latency_ms: float | None
    failure: str | None


@dataclass(frozen=True)
class StackConfig:
    compose_file: Path
    qdrant_url: str = "http://127.0.0.1:6333"
    mnemos_url: str = "http://127.0.0.1:8700"
    research_ui_url: str | None = "http://127.0.0.1:8788"
    proxy_url: str | None = "http://127.0.0.1:8790"
    qdrant_service: str = "qdrant"
    postgres_service: str = "postgres"
    mnemos_service: str = "mnemos"
    research_ui_service: str = "research-ui"
    proxy_service: str = "openwebui-proxy"
    timeout_s: float = 5.0
    require_research_ui: bool = False
    require_proxy: bool = False

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("timeout must be greater than zero")
        if not str(self.compose_file):
            raise ValueError("compose file is required")


@dataclass
class ServiceCheck:
    role: str
    compose_service: str
    required: bool
    status: str = "unknown"
    reason_code: str = "NOT_CHECKED"
    remediation: str = ""
    container_name: str | None = None
    image: str | None = None
    endpoint: str | None = None
    expected_host_port: int | None = None
    published_ports: list[int] = field(default_factory=list)
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StackReceipt:
    ok: bool
    compose_file: str
    services: dict[str, ServiceCheck]
    reason_code: str = "OK"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mnemos.local-stack-receipt.r0",
            "ok": self.ok,
            "reason_code": self.reason_code,
            "compose_file": self.compose_file,
            "services": {name: check.to_dict() for name, check in sorted(self.services.items())},
        }


class CommandRunner(Protocol):
    def run(self, args: list[str], timeout_s: float) -> CommandResult: ...


class HttpProbe(Protocol):
    def get(self, url: str, timeout_s: float) -> HttpResult: ...


class SubprocessRunner:
    def run(self, args: list[str], timeout_s: float) -> CommandResult:
        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return CommandResult(127, "", _safe_failure(exc))
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


class UrllibProbe:
    def get(self, url: str, timeout_s: float) -> HttpResult:
        started = time.perf_counter()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "mnemos-stack-verifier-r0"})
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                body = response.read(1_048_576).decode("utf-8", errors="replace")
                latency = (time.perf_counter() - started) * 1000
                status = int(response.status)
                return HttpResult(200 <= status < 300, status, body, latency, None)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            latency = (time.perf_counter() - started) * 1000
            return HttpResult(False, getattr(exc, "code", None), "", latency, _safe_failure(exc))


def _safe_failure(exc: BaseException | str) -> str:
    text = str(exc).replace("\r", " ").replace("\n", " ")
    return text[:240]


def _compose_prefix(config: StackConfig) -> list[str]:
    return ["docker", "compose", "-f", str(config.compose_file)]


def _parse_json_output(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return [json.loads(line) for line in stripped.splitlines() if line.strip()]


def _service_from_row(row: dict[str, Any]) -> str | None:
    if row.get("Service"):
        return str(row["Service"])
    labels = row.get("Labels", {})
    if isinstance(labels, dict):
        value = labels.get("com.docker.compose.service")
        return str(value) if value else None
    for label in str(labels).split(","):
        key, separator, value = label.partition("=")
        if separator and key.strip() == "com.docker.compose.service":
            return value.strip()
    return None


def _published_ports(row: dict[str, Any]) -> list[int]:
    ports: list[int] = []
    publishers = row.get("Publishers") or []
    if isinstance(publishers, list):
        for publisher in publishers:
            if isinstance(publisher, dict) and publisher.get("PublishedPort") not in (None, 0, ""):
                try:
                    ports.append(int(publisher["PublishedPort"]))
                except (TypeError, ValueError):
                    pass
    return sorted(set(ports))


def _url_port(url: str) -> int:
    parsed = urllib.parse.urlsplit(url)
    if parsed.port:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


def _join_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _mark_failure(check: ServiceCheck, reason: str, remediation: str) -> None:
    check.status = "unhealthy"
    check.reason_code = reason
    check.remediation = remediation


def _probe_http(check: ServiceCheck, http: HttpProbe, url: str, timeout_s: float) -> HttpResult | None:
    result = http.get(url, timeout_s)
    check.latency_ms = result.latency_ms
    if not result.ok:
        _mark_failure(
            check,
            "HTTP_CONNECTION_FAILED",
            f"Confirm the Compose service is running and owns host port {check.expected_host_port}; then retry {url}.",
        )
        return None
    return result


def _initial_checks(config: StackConfig) -> dict[str, ServiceCheck]:
    return {
        "qdrant": ServiceCheck("qdrant", config.qdrant_service, True, endpoint=config.qdrant_url),
        "postgres": ServiceCheck("postgres", config.postgres_service, True, expected_host_port=5432),
        "mnemos": ServiceCheck("mnemos", config.mnemos_service, True, endpoint=config.mnemos_url),
        "research-ui": ServiceCheck(
            "research-ui", config.research_ui_service, config.require_research_ui, endpoint=config.research_ui_url
        ),
        "openwebui-proxy": ServiceCheck(
            "openwebui-proxy", config.proxy_service, config.require_proxy, endpoint=config.proxy_url
        ),
    }


def run_preflight(config: StackConfig, runner: CommandRunner, http: HttpProbe) -> StackReceipt:
    """Discover and probe the configured stack without changing Docker or MNEMOS state."""
    checks = _initial_checks(config)
    for check in checks.values():
        if check.endpoint:
            check.expected_host_port = _url_port(check.endpoint)

    prefix = _compose_prefix(config)
    compose = runner.run(prefix + ["config", "--format", "json"], config.timeout_s)
    if compose.returncode != 0:
        for check in checks.values():
            _mark_failure(check, "COMPOSE_CONFIG_FAILED", "Verify Docker Compose and the selected Compose file are available.")
        return StackReceipt(False, str(config.compose_file), checks, "CONFIGURATION_ERROR")
    try:
        compose_config = _parse_json_output(compose.stdout)
        declared = compose_config.get("services", {})
    except (json.JSONDecodeError, AttributeError, TypeError):
        for check in checks.values():
            _mark_failure(check, "INVALID_COMPOSE_JSON", "Run 'docker compose config' and correct the Compose file.")
        return StackReceipt(False, str(config.compose_file), checks, "CONFIGURATION_ERROR")

    ps = runner.run(prefix + ["ps", "--format", "json"], config.timeout_s)
    try:
        rows = _parse_json_output(ps.stdout) if ps.returncode == 0 else []
        if isinstance(rows, dict):
            rows = [rows]
    except (json.JSONDecodeError, TypeError):
        rows = []
    by_service = {
        service: row
        for row in rows
        if isinstance(row, dict) and (service := _service_from_row(row)) is not None
    }

    for check in checks.values():
        service_config = declared.get(check.compose_service)
        if service_config is None:
            if not check.required and check.endpoint is None:
                check.status = "not_configured"
                check.reason_code = "OPTIONAL_NOT_CONFIGURED"
                check.remediation = "No action required; pass its URL and require flag to include this optional role."
            else:
                _mark_failure(
                    check,
                    "COMPOSE_SERVICE_MISSING",
                    f"Set the {check.role} service override to a service declared in the Compose file.",
                )
            continue
        row = by_service.get(check.compose_service)
        if row is None:
            _mark_failure(
                check,
                "CONTAINER_NOT_RUNNING",
                f"Start the {check.compose_service} service outside this read-only verifier, then retry.",
            )
            continue
        check.container_name = str(row.get("Name") or row.get("Names") or "") or None
        check.image = str(row.get("Image") or service_config.get("image") or "") or None
        check.published_ports = _published_ports(row)
        if check.expected_host_port not in check.published_ports:
            _mark_failure(
                check,
                "PORT_OWNERSHIP_MISMATCH",
                f"Publish host port {check.expected_host_port} from Compose service {check.compose_service} or override its URL.",
            )
            continue
        check.status = "discovered"
        check.reason_code = "DISCOVERED"

    qdrant_declared = declared.get(config.qdrant_service, {})
    expected_qdrant_image = "qdrant/qdrant:v1.17.1"
    if qdrant_declared.get("image") != expected_qdrant_image:
        _mark_failure(
            checks["qdrant"],
            "IMAGE_MISMATCH",
            f"Use the trial-pinned Qdrant image {expected_qdrant_image} or explicitly revise the verifier contract.",
        )

    qdrant = checks["qdrant"]
    if qdrant.status == "discovered":
        if _probe_http(qdrant, http, _join_url(config.qdrant_url, "/healthz"), config.timeout_s):
            qdrant.status, qdrant.reason_code, qdrant.remediation = "healthy", "OK", "No action required."

    postgres = checks["postgres"]
    if postgres.status == "discovered":
        pg_config = declared.get(config.postgres_service, {})
        environment = pg_config.get("environment") or {}
        if isinstance(environment, list):
            environment = dict(item.split("=", 1) for item in environment if "=" in item)
        user = str(environment.get("POSTGRES_USER") or "postgres")
        database = str(environment.get("POSTGRES_DB") or user)
        ready = runner.run(
            prefix + ["exec", "-T", config.postgres_service, "pg_isready", "-U", user, "-d", database],
            config.timeout_s,
        )
        if ready.returncode == 0:
            postgres.status, postgres.reason_code, postgres.remediation = "healthy", "OK", "No action required."
        else:
            _mark_failure(
                postgres,
                "POSTGRES_NOT_READY",
                f"Inspect readiness for Compose service {config.postgres_service}; credentials are intentionally omitted.",
            )

    mnemos = checks["mnemos"]
    if mnemos.status == "discovered":
        health_url = _join_url(config.mnemos_url, "/health")
        health = _probe_http(mnemos, http, health_url, config.timeout_s)
        if health is not None:
            try:
                health_payload = json.loads(health.body)
            except json.JSONDecodeError:
                health_payload = {}
            if isinstance(health_payload, dict) and str(health_payload.get("status", "")).lower() == "degraded":
                _mark_failure(mnemos, "SERVICE_DEGRADED", "Inspect MNEMOS health details and its Qdrant/PostgreSQL dependencies.")
            else:
                cap_url = _join_url(config.mnemos_url, "/v1/mnemos/capabilities")
                capabilities = _probe_http(mnemos, http, cap_url, config.timeout_s)
                if capabilities is not None:
                    try:
                        payload = json.loads(capabilities.body)
                    except json.JSONDecodeError:
                        _mark_failure(
                            mnemos,
                            "INVALID_CAPABILITIES_JSON",
                            f"Confirm {cap_url} returns a JSON capability document.",
                        )
                    else:
                        if not isinstance(payload, dict):
                            _mark_failure(
                                mnemos,
                                "INVALID_CAPABILITIES_JSON",
                                f"Confirm {cap_url} returns a JSON object.",
                            )
                        else:
                            mnemos.status, mnemos.reason_code, mnemos.remediation = "healthy", "OK", "No action required."

    optional_specs = (
        (checks["research-ui"], config.research_ui_url, "/"),
        (checks["openwebui-proxy"], config.proxy_url, "/health"),
    )
    for check, base_url, path in optional_specs:
        if check.status == "discovered" and base_url:
            if _probe_http(check, http, _join_url(base_url, path), config.timeout_s):
                check.status, check.reason_code, check.remediation = "healthy", "OK", "No action required."

    ok = all(check.status == "healthy" for check in checks.values() if check.required)
    return StackReceipt(ok, str(config.compose_file), checks, "OK" if ok else "REQUIRED_SERVICE_FAILURE")


def write_receipt(receipt: StackReceipt, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose-file", required=True, type=Path)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--mnemos-url", default="http://127.0.0.1:8700")
    parser.add_argument("--research-ui-url", default="http://127.0.0.1:8788")
    parser.add_argument("--openwebui-proxy-url", dest="proxy_url", default="http://127.0.0.1:8790")
    parser.add_argument("--qdrant-service", default="qdrant")
    parser.add_argument("--postgres-service", default="postgres")
    parser.add_argument("--mnemos-service", default="mnemos")
    parser.add_argument("--research-ui-service", default="research-ui")
    parser.add_argument("--openwebui-proxy-service", dest="proxy_service", default="openwebui-proxy")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--require-research-ui", action="store_true")
    parser.add_argument("--require-openwebui-proxy", dest="require_proxy", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser


def _print_receipt(receipt: StackReceipt) -> None:
    print("role                 service              container                 image                         port   status       reason")
    for role, check in receipt.services.items():
        print(
            f"{role:20} {check.compose_service:20} {(check.container_name or '-'):25} "
            f"{(check.image or '-'):29} {str(check.expected_host_port or '-'):6} {check.status:12} {check.reason_code}"
        )
        if check.status not in {"healthy", "not_configured"}:
            print(f"  remediation: {check.remediation}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = StackConfig(
            compose_file=args.compose_file,
            qdrant_url=args.qdrant_url,
            mnemos_url=args.mnemos_url,
            research_ui_url=args.research_ui_url,
            proxy_url=args.proxy_url,
            qdrant_service=args.qdrant_service,
            postgres_service=args.postgres_service,
            mnemos_service=args.mnemos_service,
            research_ui_service=args.research_ui_service,
            proxy_service=args.proxy_service,
            timeout_s=args.timeout_s,
            require_research_ui=args.require_research_ui,
            require_proxy=args.require_proxy,
        )
    except ValueError as exc:
        print(f"configuration error: {_safe_failure(exc)}", file=sys.stderr)
        return 2
    receipt = run_preflight(config, SubprocessRunner(), UrllibProbe())
    _print_receipt(receipt)
    if args.output_json:
        write_receipt(receipt, args.output_json)
    if receipt.reason_code == "CONFIGURATION_ERROR":
        return 2
    return 0 if receipt.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
