from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .canonical import artifact_id, sha256_bytes, source_uri
from .errors import ErrorCode, ProjectMemoryError
from .models import ProjectArtifact, SnapshotFile, SnapshotManifest, SourceSpan


PARSER_NAME = "python_ast_r0"
PARSER_VERSION = "1"
_ROUTE_METHODS = {"get", "post", "put", "patch", "delete", "route"}


def _module_name(path: str) -> str:
    without_suffix = path[:-3] if path.endswith(".py") else path
    return without_suffix.replace("/", ".")


def _source_lines(source: str) -> list[str]:
    lines = source.splitlines(keepends=True)
    return lines or [""]


def _slice(lines: list[str], span: SourceSpan) -> str:
    return "".join(lines[span.start_line - 1 : span.end_line])


def _artifact(
    *,
    manifest: SnapshotManifest,
    snapshot_file: SnapshotFile,
    artifact_type: str,
    qualified_name: str,
    span: SourceSpan,
    content: str,
    metadata: dict[str, Any],
) -> ProjectArtifact:
    content_hash = sha256_bytes(content.encode("utf-8"))
    identity = artifact_id(
        manifest.repo_id,
        manifest.snapshot_id,
        snapshot_file.path,
        artifact_type,
        qualified_name,
        span,
        content_hash,
    )
    return ProjectArtifact(
        artifact_id=identity,
        repo_id=manifest.repo_id,
        snapshot_id=manifest.snapshot_id,
        file_path=snapshot_file.path,
        file_hash=snapshot_file.file_hash,
        language="python",
        artifact_type=artifact_type,
        qualified_name=qualified_name,
        span=span,
        content=content,
        content_hash=content_hash,
        source_uri=source_uri(manifest.repo_id, snapshot_file.path, manifest.snapshot_id),
        parser=PARSER_NAME,
        parser_version=PARSER_VERSION,
        metadata=metadata,
    )


def _node_span(node: ast.AST, *, include_decorators: bool = False) -> SourceSpan:
    start = int(getattr(node, "lineno", 1))
    if include_decorators:
        decorators = getattr(node, "decorator_list", ())
        if decorators:
            start = min(start, *(int(item.lineno) for item in decorators))
    end = int(getattr(node, "end_lineno", start) or start)
    return SourceSpan(start, end)


def _literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def _decorator_name(node: ast.AST) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _route_metadata(node: ast.FunctionDef | ast.AsyncFunctionDef, source: str) -> dict[str, Any]:
    for decorator in node.decorator_list:
        terminal = _decorator_name(decorator).lower()
        if terminal not in _ROUTE_METHODS:
            continue
        route_path = None
        methods: list[str] = []
        if isinstance(decorator, ast.Call):
            if decorator.args:
                literal = _literal_value(decorator.args[0])
                if isinstance(literal, str):
                    route_path = literal
            for keyword in decorator.keywords:
                if keyword.arg == "methods":
                    literal = _literal_value(keyword.value)
                    if isinstance(literal, (list, tuple)):
                        methods = [str(item).upper() for item in literal if isinstance(item, str)]
        if terminal != "route":
            methods = [terminal.upper()]
        return {
            "route_detection": "heuristic",
            "detection_basis": ast.get_source_segment(source, decorator) or terminal,
            "route_path": route_path,
            "http_methods": methods,
        }
    return {}


class _Visitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        source: str,
        lines: list[str],
        manifest: SnapshotManifest,
        snapshot_file: SnapshotFile,
    ) -> None:
        self.source = source
        self.lines = lines
        self.manifest = manifest
        self.snapshot_file = snapshot_file
        self.module = _module_name(snapshot_file.path)
        self.parents: list[str] = []
        self.artifacts: list[ProjectArtifact] = []

    def _add(
        self,
        node: ast.AST,
        artifact_type: str,
        qualified_name: str,
        metadata: dict[str, Any],
        *,
        include_decorators: bool = False,
    ) -> None:
        span = _node_span(node, include_decorators=include_decorators)
        self.artifacts.append(
            _artifact(
                manifest=self.manifest,
                snapshot_file=self.snapshot_file,
                artifact_type=artifact_type,
                qualified_name=qualified_name,
                span=span,
                content=_slice(self.lines, span),
                metadata=metadata,
            )
        )

    def _qualified(self, name: str) -> str:
        return ".".join((self.module, *self.parents, name))

    def _decorators(self, node: ast.AST, owner: str) -> None:
        for index, decorator in enumerate(getattr(node, "decorator_list", ())):
            self._add(
                decorator,
                "python_decorator",
                f"{owner}@decorator[{index}]",
                {
                    "owner": owner,
                    "decorator_expression": ast.get_source_segment(self.source, decorator) or "",
                },
            )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualified = self._qualified(node.name)
        self._add(
            node,
            "python_symbol",
            qualified,
            {
                "symbol_name": node.name,
                "symbol_kind": "class",
                "parent_symbol": ".".join((self.module, *self.parents)),
            },
            include_decorators=True,
        )
        self._decorators(node, qualified)
        self.parents.append(node.name)
        self.generic_visit(node)
        self.parents.pop()

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified = self._qualified(node.name)
        is_method = bool(self.parents)
        is_test = node.name.startswith("test_")
        symbol_kind = (
            "test_method" if is_method and is_test else
            "method" if is_method else
            "test_function" if is_test else
            "function"
        )
        metadata = {
            "symbol_name": node.name,
            "symbol_kind": symbol_kind,
            "parent_symbol": ".".join((self.module, *self.parents)),
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            **_route_metadata(node, self.source),
        }
        self._add(
            node,
            "python_symbol",
            qualified,
            metadata,
            include_decorators=True,
        )
        self._decorators(node, qualified)
        self.parents.append(node.name)
        self.generic_visit(node)
        self.parents.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def visit_Import(self, node: ast.Import) -> None:
        names = [alias.name for alias in node.names]
        self._add(
            node,
            "python_import",
            f"{self.module}#import:{node.lineno}",
            {"import_kind": "import", "modules": names},
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._add(
            node,
            "python_import",
            f"{self.module}#import:{node.lineno}",
            {
                "import_kind": "from",
                "module": node.module,
                "level": node.level,
                "names": [alias.name for alias in node.names],
            },
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self.parents and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            value = _literal_value(node.value)
            if name.isupper() and value is not None:
                self._add(
                    node,
                    "python_config_constant",
                    f"{self.module}.{name}",
                    {"symbol_name": name, "symbol_kind": "configuration_constant", "literal_value": value},
                )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not self.parents and isinstance(node.target, ast.Name) and node.value is not None:
            name = node.target.id
            value = _literal_value(node.value)
            if name.isupper() and value is not None:
                self._add(
                    node,
                    "python_config_constant",
                    f"{self.module}.{name}",
                    {"symbol_name": name, "symbol_kind": "configuration_constant", "literal_value": value},
                )
        self.generic_visit(node)


def extract_python(
    project_root: Path,
    manifest: SnapshotManifest,
    snapshot_file: SnapshotFile,
) -> tuple[ProjectArtifact, ...]:
    if snapshot_file.language != "python":
        raise ProjectMemoryError(ErrorCode.PACKET_INTEGRITY_INVALID, "snapshot file is not Python")
    path = Path(project_root) / snapshot_file.path
    raw = path.read_bytes()
    if sha256_bytes(raw) != snapshot_file.file_hash:
        raise ProjectMemoryError(
            ErrorCode.PACKET_INTEGRITY_INVALID,
            "Python source no longer matches the snapshot",
        )
    try:
        source = raw.decode("utf-8", errors="strict")
        tree = ast.parse(source, filename=snapshot_file.path, type_comments=True)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ProjectMemoryError(
            ErrorCode.STRUCTURED_PARSE_INCOMPLETE,
            "Python structured parsing failed",
            details={"line": getattr(exc, "lineno", None), "category": type(exc).__name__},
        ) from exc
    lines = _source_lines(source)
    module_span = SourceSpan(1, max(1, len(lines)))
    artifacts = [
        _artifact(
            manifest=manifest,
            snapshot_file=snapshot_file,
            artifact_type="python_module",
            qualified_name=_module_name(snapshot_file.path),
            span=module_span,
            content=source,
            metadata={"symbol_kind": "module"},
        )
    ]
    visitor = _Visitor(
        source=source,
        lines=lines,
        manifest=manifest,
        snapshot_file=snapshot_file,
    )
    visitor.visit(tree)
    artifacts.extend(visitor.artifacts)
    return tuple(sorted(artifacts, key=lambda item: (item.span.start_line, item.span.end_line, item.artifact_id)))
