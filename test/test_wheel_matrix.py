"""The published matrix, as the CLI reads it, against release.yml, which is what builds it.

``fast-ulysses wheel-url`` derives an install command from
``python/fast_ulysses/wheel_matrix.json``: a base version plus a local version tag plus a python
tag. Nothing at install time can tell a tag that names a wheel from one that does not -- pip
answers "no matching distribution" either way -- so a row edited in release.yml and not here (or
the reverse) makes the tool print a command that 404s, for as long as it takes someone to
notice. That is what this file is for, and it is the only thing that checks it.

Stdlib only, no GPU and no torch: release.yml's matrix lines are flow-style YAML mappings, which
regex reads exactly, and adding a yaml dependency to run four assertions is not worth it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MATRIX = _ROOT / "python" / "fast_ulysses" / "wheel_matrix.json"
_RELEASE = _ROOT / ".github" / "workflows" / "release.yml"
_BUILD = _ROOT / "tools" / "build_wheels.sh"

pytestmark = pytest.mark.skipif(
    not _RELEASE.is_file(), reason="not a source checkout; release.yml is not here"
)


def _rows_from_release() -> list[dict[str, str]]:
    """Every `- {torch: ..., tag: ...}` line of the wheels matrix, in file order."""
    text = _RELEASE.read_text()
    start = text.index("  wheels:")
    end = text.index("\n  verify:", start)
    rows = []
    for line in text[start:end].splitlines():
        line = line.strip()
        if not line.startswith("- {") or "tag:" not in line:
            continue
        rows.append(dict(re.findall(r'(\w+):\s*"?([^",}]+)"?', line)))
    return rows


def _tag(row: dict) -> str:
    """The same derivation cli.py's _tag() makes, kept independent of it on purpose."""
    major, minor = row["torch"].split(".")[:2]
    return f"torch{major}{minor}cu{row['cuda'].replace('.', '')}"


def test_tags_match_release_matrix():
    matrix = json.loads(_MATRIX.read_text())
    release = _rows_from_release()
    assert [_tag(row) for row in matrix["rows"]] == [row["tag"] for row in release]


def test_one_pypi_row_and_it_is_the_one_release_yml_marks():
    matrix = json.loads(_MATRIX.read_text())
    tagged = [_tag(row) for row in matrix["rows"] if row.get("pypi")]
    marked = [row["tag"] for row in _rows_from_release() if row.get("pypi_set")]
    assert tagged == marked == [_tag(matrix["rows"][-1])]


def test_pythons_and_platform_tag_are_what_build_wheels_sh_emits():
    matrix = json.loads(_MATRIX.read_text())
    build = _BUILD.read_text()
    default = re.search(r'PYTHONS="\$\{PYTHONS:-([^}"]+)\}"', build)
    assert default and default.group(1).split() == matrix["pythons"]
    assert f"--plat {matrix['platform_tag']}" in build
