"""Packaging and deployment files: declared dependencies, docker-compose and the Dockerfile."""

import os
from pathlib import Path


def test_rich_is_a_declared_dependency():
    """The CLI's progress bars need rich, so it is a dependency."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as f:
        assert '"rich"' in f.read()


def test_compose_publishes_on_this_machine_and_images_are_pinned():
    """docker-compose.yml publishes the port on 127.0.0.1 only (a reverse proxy faces the network),
    and the Dockerfile's base images name their Debian release."""
    root = Path(__file__).resolve().parent.parent
    compose = (root / "docker-compose.yml").read_text()
    assert 'ports: ["127.0.0.1:8000:8000"]' in compose
    bases = [line.split()[1] for line in (root / "Dockerfile").read_text().splitlines() if line.startswith("FROM ")]
    assert bases == ["node:22-bookworm-slim", "python:3.12-slim-bookworm"]
