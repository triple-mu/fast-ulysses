#!/usr/bin/env python3
"""Emit the static PEP 503 index the release wheels are meant to be installed through.

Why an index exists at all. Every tagged wheel carries a PEP 440 local version --
``0.2.0+torch211cu130`` -- and local versions are ordered, so ``0.2.0+torch213cu130`` is the
newest ``0.2.0`` that exists. Pointing pip at a flat list of all 36 wheels therefore resolves to
the torch 2.13 build on every machine, silently, whatever torch is installed. Two things together
make the resolution correct instead:

* **One sub-index per CUDA major.** Nothing in a wheel's metadata records which CUDA it was built
  against, so it cannot be resolved -- it can only be chosen, and the only place a user can
  express the choice is the index URL. This is why download.pytorch.org is laid out as
  ``/whl/cu128/``, ``/whl/cu130/`` rather than one index with everything in it.
* **The torch pin already in each wheel** (``Requires-Dist: torch==2.11.*``). A sub-index carries
  fast-ulysses and nothing else, so pip cannot install a torch to satisfy its first choice: it
  backtracks through the candidates and stops at the one whose pin matches the torch that is
  already installed. That is what makes ``pip install fast-ulysses --index-url <sub-index>``
  land on the right wheel without the user writing the local version out by hand.

The per-row indexes below take that second axis out of the resolver too, for a user who knows
their torch minor and would rather have both choices recorded in the URL.

Layout, given ``--out site``::

    site/.nojekyll
    site/index.html                             # human landing page, lists the sub-indexes
    site/whl/cu12/index.html                    # PEP 503 root:    one anchor per project
    site/whl/cu12/fast-ulysses/index.html       # PEP 503 project: one anchor per file
    site/whl/cu128/... , site/whl/cu129/...     # same set as cu12
    site/whl/cu13/... , site/whl/cu130/...
    site/whl/torch210cu128/...                  # one row: 4 wheels, one per CPython
    site/whl/torch210cu130/... (8 in all)

Three kinds of name, and every wheel is in three of them:

``cu12`` / ``cu13``
    The CUDA **major**, which is the axis that has to match and cannot be resolved. The torch
    minor is left to the pin, so this is the one URL a user can pick knowing only their CUDA.
``cu128`` / ``cu129`` / ``cu130``
    Aliases of their major, one per ``cuXXX`` appearing in the release tags. A user reads that
    suffix off the index they installed torch from and will try it here first; the CUDA minor
    carries no constraint of its own, so the alias holds the whole major.
``torch210cu128`` ... (the eight matrix tags)
    One row each: four wheels, one per CPython, nothing to resolve and nothing to compare. It is
    the form that still pins when the user has to add ``--extra-index-url`` for some other
    package. Keyed on the torch minor and the CUDA MAJOR, not on the exact tag: 2.12 moved from
    cu128 to cu129 between releases, and ``whl/torch212cu128/`` has to keep receiving that row
    or every URL written down against it silently stops at the release it was written in.

Anchors point at the GitHub Release assets -- PEP 503 puts no constraint on where the files live
-- and carry ``#sha256=``, which pip enforces. The output tree is MERGED into whatever is already
in ``--out``: a v0.3.0 publish must leave v0.2.0's links in place, or every pin written down
against them breaks.

Wheels with no local version are skipped: that is the set that goes to PyPI, and it has no CUDA
tag to file it under.

Stdlib only, like tools/check_wheel.py, so it runs in the release container and on a laptop.

Usage:
    tools/make_index.py --out site \\
        --release-url https://github.com/triple-mu/fast-ulysses/releases/download/v0.2.0 \\
        --site-url https://triple-mu.github.io/fast-ulysses \\
        wheelhouse/
"""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import sys
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

# The project as it is published, and the PEP 503 normalisation of it (which is what the
# directory has to be named). The wheel FILENAME uses fast_ulysses; that is unrelated.
PROJECT = "fast-ulysses"
NORMALIZED = re.sub(r"[-_.]+", "-", PROJECT).lower()
WHEEL_PREFIX = PROJECT.replace("-", "_") + "-"

# The local version release.yml's matrix puts in the filename, e.g. torch213cu130. Same shape
# check_wheel.py's assertion 11 parses; here only the CUDA part is used.
TAG_RE = re.compile(r"^torch(\d)(\d+)cu(\d{2})(\d+)$")

_PAGE = """<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="pypi:repository-version" content="1.0">
    <title>{title}</title>
  </head>
  <body>
    <h1>{title}</h1>
{body}
  </body>
</html>
"""


@dataclass(frozen=True)
class Wheel:
    filename: str
    version: str  # "0.2.0+torch213cu130"
    tag: str  # "torch213cu130"
    cuda_major: str  # "13"
    cuda_tag: str  # "cu130"
    sha256: str
    requires_python: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _requires_python(path: Path) -> str | None:
    """The wheel's own Requires-Python, for the data-requires-python attribute.

    Advisory (PEP 503 says installers SHOULD honour it) and redundant with the cp3XX ABI tag
    pip already filters on, but it is free and it makes the page readable.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            metadata = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise SystemExit(f"make_index: {path.name} has {len(metadata)} .dist-info/METADATA")
            text = zf.read(metadata[0]).decode()
    except zipfile.BadZipFile:
        raise SystemExit(f"make_index: {path.name} is not a readable wheel (zip)")
    match = re.search(r"^Requires-Python:\s*(.+)$", text, re.M)
    return match.group(1).strip() if match else None


def read_wheel(path: Path) -> Wheel | None:
    """Parse one wheel; None for the untagged PyPI set, which this index does not carry."""
    name = path.name
    if not name.endswith(".whl"):
        raise SystemExit(f"make_index: {name} is not a wheel")
    parts = name[: -len(".whl")].split("-")
    if len(parts) < 5 or not name.startswith(WHEEL_PREFIX):
        raise SystemExit(f"make_index: {name} is not a {PROJECT} wheel filename")

    _, _, local = parts[1].partition("+")
    if not local:
        return None
    tag = TAG_RE.match(local)
    if not tag:
        # Loud, not skipped. A wheel silently absent from the index is the failure this whole
        # file exists to prevent, and a tag that does not parse is how it would happen.
        raise SystemExit(
            f"make_index: local version {local!r} of {name} is not "
            "torch<major><minor>cu<major><minor>"
        )
    return Wheel(
        filename=name,
        version=parts[1],
        tag=local,
        cuda_major=tag.group(3),
        cuda_tag=f"cu{tag.group(3)}{tag.group(4)}",
        sha256=_sha256(path),
        requires_python=_requires_python(path),
    )


class _LinkParser(HTMLParser):
    """Anchors of an already-published project page, as {filename: attributes}.

    Merging happens at this level rather than by keeping a manifest: the published page IS the
    state, so there is nothing that can drift out of sync with it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.links: dict[str, dict[str, str]] = {}
        self._attrs: dict[str, str] | None = None
        self._text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._attrs = {k: v for k, v in attrs if v is not None}
            self._text = ""

    def handle_data(self, data: str) -> None:
        if self._attrs is not None:
            self._text += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._attrs is not None:
            name = self._text.strip()
            if name:
                self.links[name] = self._attrs
            self._attrs = None


def read_existing(page: Path) -> dict[str, dict[str, str]]:
    if not page.is_file():
        return {}
    parser = _LinkParser()
    parser.feed(page.read_text(encoding="utf-8"))
    parser.close()
    return parser.links


def _attrs(wheel: Wheel, release_url: str) -> dict[str, str]:
    # quote() percent-encodes the "+" of the local version, which is what GitHub's own
    # browser_download_url and download.pytorch.org both emit.
    href = f"{release_url.rstrip('/')}/{quote(wheel.filename)}#sha256={wheel.sha256}"
    attrs = {"href": href}
    if wheel.requires_python:
        attrs["data-requires-python"] = wheel.requires_python
    return attrs


def _anchor(text: str, attrs: dict[str, str]) -> str:
    order = ["href"] + sorted(k for k in attrs if k != "href")
    rendered = " ".join(f'{k}="{html.escape(attrs[k], quote=True)}"' for k in order)
    return f"    <a {rendered}>{html.escape(text)}</a><br />"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _render_project(links: dict[str, dict[str, str]]) -> str:
    body = "\n".join(_anchor(name, links[name]) for name in sorted(links))
    return _PAGE.format(title=f"Links for {PROJECT}", body=body)


def _render_root() -> str:
    return _PAGE.format(
        title=f"Simple index for {PROJECT}",
        body=_anchor(PROJECT, {"href": f"{NORMALIZED}/"}),
    )


def _describe(name: str) -> str:
    """One line per sub-index, derived from its name -- there is no table to fall out of date."""
    if re.fullmatch(r"cu\d{2}", name):
        return f"every wheel built against CUDA {name[2:]}; the torch minor is left to the pin"
    if re.fullmatch(r"cu\d{3,}", name):
        return f"alias of cu{name[2:4]}, same wheels"
    row = TAG_RE.match(name)
    if row:
        # Not "CUDA <major>.<minor> only": the row keeps this URL when its CUDA minor moves.
        return (
            f"torch {row.group(1)}.{row.group(2)}, CUDA {row.group(3)} only: "
            "one row, four wheels per release, one per CPython"
        )
    return "unrecognised"


def _render_landing(names: list[str], site_url: str, example: str) -> str:
    site = site_url.rstrip("/")
    majors = [n for n in names if re.fullmatch(r"cu\d{2}", n)]
    commands = "\n".join(
        f"pip install {PROJECT} --index-url {site}/whl/{name}/"
        f"    # torch built against CUDA {name[2:]}"
        for name in majors
    )
    rows = "\n".join(
        f'      <li><a href="whl/{name}/">{site}/whl/{name}/</a> &mdash; {_describe(name)}</li>'
        for name in names
    )
    body = f"""    <p>
      PEP 503 indexes of the {PROJECT} release wheels. Pick by the CUDA your torch was built
      against (<code>python -c "import torch; print(torch.version.cuda)"</code>); the torch pin
      inside each wheel then selects the build for the torch you have.
    </p>
    <pre>{commands}</pre>
    <p>
      <code>--index-url</code> and nothing else. pip merges every index it is given and takes the
      highest version across all of them, and the newest local version here is always the newest
      torch &mdash; so adding <code>--extra-index-url https://pypi.org/simple</code> makes pip
      install that wheel and <em>upgrade your torch to match it</em>. These indexes carry no torch
      of their own, which is what leaves pip no choice but to fall back to the wheel pinned to the
      torch you already have. Install torch first, from wherever you normally do.
    </p>
    <p>
      If something else in the environment must come from PyPI anyway, use a per-row index below
      &mdash; only one row is in it, so nothing else can be selected &mdash; or write the version
      out in full: <code>pip install "{PROJECT}=={example}"</code> resolves to exactly one file,
      under pip and under uv alike. That has to be <em>your</em> row's tag, not this one: any
      other row pins a torch you do not have, and in a command that can reach PyPI pip satisfies
      that pin by moving your torch, not by failing. The tag for each row is in the list below.
    </p>
    <ul>
{rows}
    </ul>"""
    return _PAGE.format(title=f"{PROJECT} wheel index", body=body)


def _row_key(tag: str) -> str:
    """A row's identity across releases: its torch minor and its CUDA MAJOR.

    The CUDA minor is deliberately not part of it. torch 2.12's row moved from cu128 to cu129
    between two releases, and both names have to keep meaning the same row.
    """
    row = TAG_RE.match(tag)
    if not row:  # unreachable for a Wheel: read_wheel() already refused anything else
        raise SystemExit(f"make_index: {tag!r} is not a row tag")
    return f"torch{row.group(1)}{row.group(2)}cu{row.group(3)}"


def partition(wheels: list[Wheel], out: Path) -> dict[str, list[Wheel]]:
    """Which sub-index holds which wheels. See this file's header for the three kinds of name."""
    aliases: dict[str, set[str]] = {}
    rows: dict[str, set[str]] = {}
    for wheel in wheels:
        aliases.setdefault(wheel.cuda_major, {f"cu{wheel.cuda_major}"}).add(wheel.cuda_tag)
        rows.setdefault(_row_key(wheel.tag), set()).add(wheel.tag)

    # A name that already exists keeps receiving every wheel it still describes, even after the
    # matrix stops building that exact CUDA minor -- cu128 keeps getting the CUDA 12 wheels, and
    # torch212cu128 keeps getting the torch 2.12 CUDA 12 wheels. Otherwise the release where the
    # row moves from cu128 to cu129 leaves both of those frozen at the previous version, and a
    # user pinned to either URL stops seeing new releases without anything saying so. Matching is
    # on the CUDA major both times: nothing here may put a CUDA 13 wheel behind a cu12 name.
    for existing in sorted((out / "whl").glob("cu*")):
        alias = re.fullmatch(r"cu(\d{2})\d*", existing.name)
        if alias and alias.group(1) in aliases:
            aliases[alias.group(1)].add(existing.name)
    for existing in sorted((out / "whl").glob("torch*")):
        if TAG_RE.match(existing.name) and _row_key(existing.name) in rows:
            rows[_row_key(existing.name)].add(existing.name)

    groups: dict[str, list[Wheel]] = {}
    for wheel in wheels:
        for name in aliases[wheel.cuda_major] | rows[_row_key(wheel.tag)]:
            groups.setdefault(name, []).append(wheel)
    return groups


def build(wheels: list[Wheel], out: Path, release_url: str, site_url: str) -> None:
    for name, group in sorted(partition(wheels, out).items()):
        page = out / "whl" / name / NORMALIZED / "index.html"
        links = read_existing(page)
        before = len(links)
        for wheel in group:
            links[wheel.filename] = _attrs(wheel, release_url)
        _write(page, _render_project(links))
        _write(out / "whl" / name / "index.html", _render_root())
        print(f"whl/{name}/{NORMALIZED}/: {len(links)} links, {len(links) - before} new")

    # Every sub-index in the tree, including ones this release did not touch -- an older CUDA
    # row that no longer builds must stay reachable, or the pins against it stop resolving.
    published = sorted(
        d.name for d in (out / "whl").iterdir() if (d / NORMALIZED / "index.html").is_file()
    )
    # One release, one base version, so this compares local labels only -- any of them would do
    # as the landing page's example of a fully written-out specifier.
    example = max(wheel.version for wheel in wheels)
    _write(out / "index.html", _render_landing(published, site_url, example))
    # GitHub Pages runs Jekyll otherwise, which drops paths beginning with "_" and adds a build
    # step this tree has no use for.
    _write(out / ".nojekyll", "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheels", nargs="+", type=Path, help="wheel files, or directories to scan")
    parser.add_argument("--out", required=True, type=Path, help="tree to write, merged into")
    parser.add_argument(
        "--release-url",
        required=True,
        help="directory URL the wheels are served from, "
        "e.g. https://github.com/o/r/releases/download/v0.2.0",
    )
    parser.add_argument(
        "--site-url", required=True, help="where this tree will be served, for the landing page"
    )
    args = parser.parse_args(argv)

    paths: list[Path] = []
    for given in args.wheels:
        if given.is_dir():
            # Recursive: the release job's download-artifact lands them one directory per row.
            paths += sorted(given.rglob("*.whl"))
        elif given.is_file():
            paths.append(given)
        else:
            raise SystemExit(f"make_index: no such file or directory: {given}")
    if not paths:
        raise SystemExit(f"make_index: no wheels found in {[str(p) for p in args.wheels]}")

    wheels, skipped = [], []
    for path in paths:
        wheel = read_wheel(path)
        if wheel is None:
            skipped.append(path.name)
        else:
            wheels.append(wheel)
    if not wheels:
        raise SystemExit(f"make_index: none of the {len(paths)} wheels carries a local version")

    build(wheels, args.out, args.release_url, args.site_url)
    if skipped:
        print(f"skipped {len(skipped)} wheel(s) with no local version (the PyPI set)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
