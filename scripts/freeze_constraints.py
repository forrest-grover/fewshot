"""Print constraints.txt: exact versions of fewshot's dependencies (with the dev and siglip extras),
and theirs, as installed in this environment. Usage, from the repo root, in an environment where
the checks pass: python scripts/freeze_constraints.py > constraints.txt

Local version labels are dropped (torch 2.11.0+cpu is pinned as 2.11.0, which pip matches with the
CPU, CUDA and macOS builds alike), and CUDA-only packages that a GPU build of torch pulls in are left
out, so the file works with the CPU wheel index too.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import re
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT, EXTRAS = "fewshot", ("dev", "siglip")
CUDA_ONLY = re.compile(r"nvidia-.*|cuda-.*|triton")
# Installed packages only: a stale fewshot.egg-info in the working directory isn't what is installed.
SITE = [p for p in sys.path if p and os.path.abspath(p) != os.getcwd()]


def distribution(name: str) -> md.Distribution:
    """The installed distribution called ``name``."""
    for dist in md.distributions(name=name, path=SITE):
        return dist
    raise md.PackageNotFoundError(name)


def requirements(dist: md.Distribution, extras: tuple[str, ...]) -> list[Requirement]:
    """The requirements of ``dist`` that apply here with ``extras``."""
    out = []
    for line in dist.requires or []:
        req = Requirement(line)
        envs = [{"extra": e} for e in ("", *extras)]
        if req.marker is None or any(req.marker.evaluate(env) for env in envs):
            out.append(req)
    return out


def main() -> None:
    pins: dict[str, str] = {}
    todo = [(ROOT, EXTRAS)]
    seen: set[tuple[str, tuple[str, ...]]] = set()
    while todo:
        name, extras = todo.pop()
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        try:
            dist = distribution(name)
        except md.PackageNotFoundError:  # optional on this platform (say uvloop on Windows)
            continue
        key = canonicalize_name(dist.metadata["Name"])
        if key != ROOT:
            pins[key] = dist.version.split("+", 1)[0]
        for req in requirements(dist, extras):
            if not CUDA_ONLY.fullmatch(canonicalize_name(req.name)):
                todo.append((req.name, tuple(sorted(req.extras))))
    print("# Exact versions the checks pass with; see \"Dependencies\" in README.md. Regenerate with")
    print("#   python scripts/freeze_constraints.py > constraints.txt")
    for key in sorted(pins):
        print(f"{key}=={pins[key]}")


if __name__ == "__main__":
    main()
