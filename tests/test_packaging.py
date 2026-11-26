"""Packaging and deployment files: declared dependencies, docker-compose and the Dockerfile."""

import ast
import importlib.metadata
import os
import re
import sys
from pathlib import Path

import pytest


def test_rich_is_a_declared_dependency():
    """The CLI's progress bars need rich, so it is a dependency."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as f:
        assert '"rich"' in f.read()


def test_scripts_import_only_declared_packages():
    """What scripts/ import is the standard library or a declared dependency (the dev extra counts)."""
    tomllib = pytest.importorskip("tomllib")
    root = Path(__file__).resolve().parent.parent
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    declared = {re.sub(r"[-_.]+", "-", re.split(r"[\[<>=!~; ]", d, maxsplit=1)[0]).lower()
                for d in project["dependencies"] + project["optional-dependencies"]["dev"]}
    providers = importlib.metadata.packages_distributions()
    for script in (root / "scripts").glob("*.py"):
        for node in ast.walk(ast.parse(script.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else [])
            for top in {n.split(".")[0] for n in names} - set(sys.stdlib_module_names) - {"fewshot"}:
                dists = {re.sub(r"[-_.]+", "-", d).lower() for d in providers.get(top, [top])}
                assert dists & declared, f"{script.name} imports {top}, which pyproject.toml doesn't declare"


def test_compose_publishes_on_this_machine_and_images_are_pinned():
    """docker-compose.yml publishes the port on 127.0.0.1 only (a reverse proxy faces the network),
    and the Dockerfile's base images name their Debian release."""
    root = Path(__file__).resolve().parent.parent
    compose = (root / "docker-compose.yml").read_text()
    assert 'ports: ["127.0.0.1:8000:8000"]' in compose
    bases = [line.split()[1] for line in (root / "Dockerfile").read_text().splitlines() if line.startswith("FROM ")]
    assert bases == ["node:22-bookworm-slim", "python:3.12-slim-bookworm"]


def test_constraints_pin_every_dependency():
    """constraints.txt pins each dependency pyproject.toml declares (extras too) to one exact version,
    without local labels (so the CPU and CUDA builds of torch both match) or CUDA-only packages;
    CI and the Dockerfile install with it."""
    tomllib = pytest.importorskip("tomllib")
    root = Path(__file__).resolve().parent.parent
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    declared = project["dependencies"] + [d for extra in project["optional-dependencies"].values() for d in extra]
    pins = {}
    for line in (root / "constraints.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            assert re.fullmatch(r"[0-9][0-9a-z.]*", version), line
            pins[name] = version
    assert not [n for n in pins if re.fullmatch(r"nvidia-.*|cuda-.*|triton", n)]
    for dep in declared:
        name = re.split(r"[\[<>=!~; ]", dep, maxsplit=1)[0]
        assert re.sub(r"[-_.]+", "-", name).lower() in pins, dep
    assert "-c constraints.txt" in (root / "Dockerfile").read_text()
    assert "-c constraints.txt" in (root / ".github" / "workflows" / "ci.yml").read_text()


def test_tokens_in_workflows_and_docs_would_be_accepted():
    """Every token set literally in CI, compose or the README passes the server's strength rule
    (a weak one makes the server refuse to start, as the Docker CI probe once did)."""
    from fewshot.server_settings import token_problem  # pylint: disable=import-outside-toplevel

    root = Path(__file__).resolve().parent.parent
    files = [*root.glob(".github/workflows/*.yml"), root / "docker-compose.yml", root / "README.md"]
    found = [(f.name, m) for f in files
             for m in re.findall(r"FEWSHOT_TOKEN=([^\s\"'$`)]+)", f.read_text())]
    for name, token in found:
        assert token_problem(token) is None, f"{name} sets FEWSHOT_TOKEN={token}, which the server refuses"
