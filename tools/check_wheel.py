#!/usr/bin/env python3
"""Prove a repaired wheel is what we think it is, before anyone can install it.

Everything checked here is invisible at build time and unfixable after upload: an RPATH still
pointing at the build host, a silently dropped GPU architecture, a glibc symbol newer than
manylinux_2_28 allows, an extra DT_NEEDED that auditwheel would have vendored because the
library happened to exist on the builder. Each one produces a wheel that installs cleanly and
then fails on someone else's machine, so the only place to catch them is here, on the artifact
that actually ships.

Run this on the REPAIRED wheel, never the raw one: ``auditwheel repair`` has been observed to
rewrite DT_RUNPATH even on a repair that vendors nothing, so a raw wheel that passes says
nothing about the wheel that ships. Assertion 5 (byte-exact RUNPATH) is the real gate and only
means anything after repair.

Stdlib only and no GPU required: this has to run in the release job, inside the build
container, and on a laptop, so it shells out to readelf/objdump/cuobjdump instead of importing
anything.

Usage:
    tools/check_wheel.py --cuda-major 13 --arch 80,90,100,120 wheelhouse/*.whl
    tools/check_wheel.py --print-excludes      # the auditwheel exclude list, one flag/line
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# The RUNPATH the extension must carry, byte for byte. torch's lib directory is the only thing
# it links that is not on a default search path, and it is $ORIGIN-relative so the wheel works
# from any site-packages.
EXPECTED_RUNPATH = "$ORIGIN/../torch/lib"

# Every one of these must be present, and anything outside REQUIRED | OPTIONAL fails, so a NEW
# external dependency is as loud as a missing one. libcudart.so.<major> is added from
# --cuda-major: it is what the extension calls, and it resolves at runtime out of the nvidia
# wheels torch depends on. libcuda.so.1 is deliberately absent: the extension reaches the driver
# only through cudart, and if that ever changes the wheel stops being installable on a driverless
# machine, which we want to hear about from the release job.
REQUIRED_NEEDED = {
    "libtorch.so",
    "libc10.so",
    "libc10_cuda.so",
    "libtorch_python.so",
    "libtorch_cpu.so",
    "libtorch_cuda.so",
    "libstdc++.so.6",
    "libgcc_s.so.1",
    "libc.so.6",
}

# Allowed but not required, because whether they appear is a property of the toolchain rather
# than of this code: glibc split libpthread/libdl out below 2.34 and folded them back into libc
# at 2.34, and the loader is named explicitly only by some linker configurations.
# libnvrtc.so.<major> is added from --cuda-major and belongs to the same class: it is on the link
# line through TORCH_LIBRARIES, but nothing here calls an nvrtc symbol, so whether it becomes a
# DT_NEEDED is decided by the builder's --as-needed default and not by this code.
OPTIONAL_NEEDED = {
    "libpthread.so.0",
    "libdl.so.2",
    "libm.so.6",
    "librt.so.1",
    "ld-linux-x86-64.so.2",
}

# Exact SONAMEs, not globs, and kept here rather than in the build script so the release job
# and CI cannot drift apart. The point of the list is that a NEW external dependency is not
# on it: auditwheel would then vendor it into fast_ulysses.libs/ and assertion 2 fails.
AUDITWHEEL_EXCLUDES = [
    "libtorch.so",
    "libtorch_cpu.so",
    "libtorch_cuda.so",
    "libtorch_python.so",
    "libc10.so",
    "libc10_cuda.so",
    "libcudart.so.12",
    "libcudart.so.13",
    "libnvrtc.so.12",
    "libnvrtc.so.13",
]

MAX_GLIBC = (2, 28)  # manylinux_2_28
MAX_GLIBCXX = (3, 4, 25)  # the GCC 9 / devtoolset-9 ABI manylinux_2_28 guarantees
REQUIRED_PLATFORM = "manylinux_2_28_x86_64"
# Substrings that only ever appear in a path that belongs to the machine that did the build.
BUILD_HOST_MARKERS = ("/opt/", "/tmp/", "/home/", "site-packages")


def _run(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit(f"check_wheel: {cmd[0]} not found; it is required to inspect the .so")
    if proc.returncode != 0:
        raise SystemExit(f"check_wheel: `{' '.join(cmd)}` failed:\n{proc.stderr.strip()}")
    return proc.stdout


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _parse_filename(wheel: Path) -> tuple[str, str, str]:
    """(version, abi tag, platform tag). Wheel names are dash-separated and the version never
    contains a dash, so a plain split is exact."""
    parts = wheel.name[: -len(".whl")].split("-")
    if len(parts) < 5:
        raise SystemExit(f"check_wheel: {wheel.name} is not a valid wheel filename")
    return parts[1], parts[-2], parts[-1]


def _dynamic(so: Path) -> tuple[list[str], list[str], list[str]]:
    """(NEEDED, DT_RPATH values, DT_RUNPATH values) as they appear in the dynamic section."""
    needed: list[str] = []
    rpath: list[str] = []
    runpath: list[str] = []
    bucket = {"NEEDED": needed, "RPATH": rpath, "RUNPATH": runpath}
    for line in _run(["readelf", "-d", str(so)]).splitlines():
        m = re.search(r"\((NEEDED|RPATH|RUNPATH)\)\s+.*\[(.*)\]", line)
        if m:
            bucket[m.group(1)].append(m.group(2))
    return needed, rpath, runpath


def _read_build_meta(zf: zipfile.ZipFile) -> dict[str, object]:
    """The generated fast_ulysses/_build_meta.py, parsed without importing it."""
    try:
        text = zf.read("fast_ulysses/_build_meta.py").decode()
    except KeyError:
        raise SystemExit("check_wheel: fast_ulysses/_build_meta.py missing from the wheel")
    meta: dict[str, object] = {}
    for line in text.splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        meta[key.strip()] = ast.literal_eval(value.strip())
    return meta


def _check_archive(wheel: Path, zf: zipfile.ZipFile, abi: str, platform: str) -> list[str]:
    """Assertions 1, 2, 9 -- what the zip itself says."""
    problems = []
    names = zf.namelist()

    # 1. exactly one extension, at the expected path, built for the wheel's own ABI
    sos = [n for n in names if n.endswith(".so")]
    expected_so = f"fast_ulysses/_C.cpython-{abi[2:]}-x86_64-linux-gnu.so"
    if len(sos) != 1:
        problems.append(f"expected exactly one .so, found {len(sos)}: {sos}")
    elif sos[0] != expected_so:
        problems.append(f"extension is {sos[0]}, expected {expected_so} for wheel ABI tag {abi}")

    # 2. auditwheel vendored nothing
    vendored = sorted({n for n in names if n.startswith("fast_ulysses.libs/")})
    if vendored:
        problems.append(
            f"auditwheel vendored libraries into fast_ulysses.libs/: {vendored}. "
            "A dependency that should have been excluded was not, or a new one appeared."
        )

    # 9. platform tag. auditwheel writes a compressed tag SET when the wheel turns out to satisfy
    # an older policy than the one requested, e.g. manylinux_2_24_x86_64.manylinux_2_28_x86_64.
    # That is a wider audience, not a defect, so require the tag we asked for to be present and
    # reject anything that is not a manylinux x86_64 tag at or below it -- which still catches an
    # unrepaired linux_x86_64 wheel and a policy newer than our floor.
    tags = platform.split(".")
    if REQUIRED_PLATFORM not in tags:
        problems.append(f"platform tags are {tags}, none of which is {REQUIRED_PLATFORM}")
    for tag in tags:
        m = re.fullmatch(r"manylinux_2_(\d+)_x86_64", tag)
        if not m:
            problems.append(f"unexpected platform tag {tag!r}")
        elif int(m.group(1)) > MAX_GLIBC[1]:
            problems.append(f"platform tag {tag!r} needs a newer glibc than manylinux_2_28")
    return problems


def _check_metadata(zf: zipfile.ZipFile, version: str, abi: str, cuda_major: str, meta: dict):
    """Assertion 10 -- Requires-Dist and Version, cross-checked against the build's own facts."""
    problems = []
    dist_info = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
    if len(dist_info) != 1:
        return [f"expected one .dist-info/METADATA, found {dist_info}"]
    text = zf.read(dist_info[0]).decode()

    declared = re.findall(r"^Version:\s*(.+)$", text, re.M)
    if declared != [version]:
        problems.append(
            f"METADATA Version: {declared} does not match the filename version {version}"
        )

    requires = [line.strip() for line in re.findall(r"^Requires-Dist:\s*(.+)$", text, re.M)]
    expected_torch = f"torch=={meta.get('TORCH_ABI', '')}.*"
    if expected_torch not in requires:
        problems.append(f"Requires-Dist has no {expected_torch!r}; it declares {requires}")

    # The wheel's own metadata has to agree with the container it was built in. A cu13 torch
    # built in a CUDA 12 image links libcudart.so.12 and then dies on import against a cu13
    # runtime, and nothing downstream of here can tell.
    built_cuda = str(meta.get("CUDA_VERSION", "")).split(".")[0]
    if built_cuda != cuda_major:
        problems.append(
            f"_build_meta CUDA_VERSION is {meta.get('CUDA_VERSION')} (major {built_cuda}) "
            f"but this wheel is being checked as CUDA {cuda_major}"
        )
    if meta.get("PYTHON_ABI") != abi:
        problems.append(
            f"_build_meta PYTHON_ABI is {meta.get('PYTHON_ABI')}, wheel ABI tag is {abi}"
        )
    if meta.get("VERSION") != version:
        problems.append(f"_build_meta VERSION is {meta.get('VERSION')}, filename says {version}")
    return problems


def _check_wheel_tag(version: str, meta: dict) -> list[str]:
    """Assertion 11 -- the local version in the filename describes the build it names.

    release.yml's matrix carries `tag:` as free text, and it is the only thing a user reads when
    picking a wheel off the release page. Every other assertion here reads METADATA, so a row
    whose tag says torch212cu130 while its torch is 2.11 passes all of them and ships a wheel
    whose filename lies. Parse the tag and put it back against the facts.
    """
    problems = []
    _, _, local = version.partition("+")
    tag = str(meta.get("WHEEL_TAG", ""))
    if tag != local:
        return [f"_build_meta WHEEL_TAG is {tag!r}, the filename's local version is {local!r}"]
    if not tag:
        # The PyPI set, built with LOCAL_VERSION empty. Nothing to cross-check.
        return problems

    m = re.fullmatch(r"torch(\d)(\d+)cu(\d{2})(\d+)", tag)
    if not m:
        return [f"WHEEL_TAG {tag!r} is not of the form torch<major><minor>cu<major><minor>"]
    tag_torch = f"{m.group(1)}.{m.group(2)}"
    tag_cuda = f"{m.group(3)}.{m.group(4)}"
    if tag_torch != meta.get("TORCH_ABI"):
        problems.append(
            f"WHEEL_TAG {tag!r} claims torch {tag_torch}, _build_meta says {meta.get('TORCH_ABI')}"
        )
    if tag_cuda != str(meta.get("CUDA_VERSION")):
        problems.append(
            f"WHEEL_TAG {tag!r} claims CUDA {tag_cuda}, _build_meta says {meta.get('CUDA_VERSION')}"
        )
    return problems


def _check_so(so: Path, cuda_major: str, arches: list[str]) -> list[str]:
    """Assertions 3-8 -- what the linker and the compiler actually produced."""
    problems = []
    needed, rpath, runpath = _dynamic(so)

    # 3. one RUNPATH, no RPATH (RPATH is not overridable by LD_LIBRARY_PATH and is ignored by
    # some loaders when RUNPATH is also present, so "both" is never what we meant)
    if rpath:
        problems.append(f"extension has DT_RPATH {rpath}; only DT_RUNPATH is allowed")
    if len(runpath) != 1:
        problems.append(f"expected exactly one DT_RUNPATH, found {len(runpath)}: {runpath}")

    if runpath:
        entries = runpath[0].split(":")
        # 4. nothing absolute and nothing from the build host
        for entry in entries:
            if not entry.startswith("$ORIGIN"):
                problems.append(f"RUNPATH entry {entry!r} is not $ORIGIN-relative")
            for marker in BUILD_HOST_MARKERS:
                if marker in entry:
                    problems.append(f"RUNPATH entry {entry!r} leaks the build host ({marker})")
        # 5. and it is exactly the string we expect
        if runpath[0] != EXPECTED_RUNPATH:
            problems.append(f"RUNPATH is\n  {runpath[0]}\nexpected\n  {EXPECTED_RUNPATH}")

    # 6. every required entry present, and nothing beyond required | optional
    required = REQUIRED_NEEDED | {f"libcudart.so.{cuda_major}"}
    optional = OPTIONAL_NEEDED | {f"libnvrtc.so.{cuda_major}"}
    got = set(needed)
    missing = sorted(required - got)
    extra = sorted(got - required - optional)
    if missing or extra:
        problems.append(f"DT_NEEDED mismatch: missing {missing}, unexpected {extra}")

    # 7. symbol versions within what manylinux_2_28 guarantees
    dynsyms = _run(["objdump", "-T", str(so)])
    for label, pattern, ceiling in (
        ("GLIBC", r"GLIBC_(\d+(?:\.\d+)*)", MAX_GLIBC),
        ("GLIBCXX", r"GLIBCXX_(\d+(?:\.\d+)*)", MAX_GLIBCXX),
    ):
        found = {_version_tuple(v) for v in re.findall(pattern, dynsyms)}
        too_new = sorted(v for v in found if v > ceiling)
        if too_new:
            printable = ", ".join(f"{label}_" + ".".join(map(str, v)) for v in too_new)
            problems.append(
                f"requires {printable}, above the {label}_"
                f"{'.'.join(map(str, ceiling))} that manylinux_2_28 guarantees"
            )

    # 8. every requested architecture is really in the fatbin. A dropped -gencode is otherwise
    # silent until someone runs on that GPU, and this needs no GPU to catch.
    elf = _run(["cuobjdump", "--list-elf", str(so)])
    got_arches = set(re.findall(r"sm_(\w+)", elf))
    want_arches = set(arches)
    if got_arches != want_arches:
        problems.append(
            f"embedded SASS is for sm_{sorted(got_arches)}, expected sm_{sorted(want_arches)}"
        )
    return problems


def check_wheel(wheel: Path, cuda_major: str | None, arches: list[str]) -> list[str]:
    version, abi, platform = _parse_filename(wheel)
    with zipfile.ZipFile(wheel) as zf:
        meta = _read_build_meta(zf)
        if cuda_major is None:
            # Not passed: take it from the build's own record of the torch it linked against.
            # _check_metadata still cross-checks that against the .so's libcudart.
            cuda_major = str(meta.get("CUDA_VERSION", "")).split(".")[0]
            if not cuda_major:
                raise SystemExit(
                    f"check_wheel: {wheel.name} has no usable _build_meta CUDA_VERSION; "
                    "pass --cuda-major"
                )
        problems = _check_archive(wheel, zf, abi, platform)
        problems += _check_metadata(zf, version, abi, cuda_major, meta)
        problems += _check_wheel_tag(version, meta)
        sos = [n for n in zf.namelist() if n.endswith(".so")]
        if not sos:
            return problems
        with tempfile.TemporaryDirectory() as tmp:
            so = Path(zf.extract(sos[0], tmp))
            problems += _check_so(so, cuda_major, arches)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheels", nargs="*", type=Path, help="repaired wheel(s) to check")
    parser.add_argument(
        "--cuda-major",
        help="CUDA major the wheel targets; default: read from the wheel's _build_meta",
    )
    parser.add_argument(
        "--arch",
        default="80,90,100,120",
        help="comma-separated SM architectures the fatbin must contain exactly",
    )
    parser.add_argument(
        "--print-excludes",
        action="store_true",
        help="print the auditwheel --exclude flags (one per line) and exit",
    )
    args = parser.parse_args(argv)

    if args.print_excludes:
        for soname in AUDITWHEEL_EXCLUDES:
            print(f"--exclude={soname}")
        return 0
    if not args.wheels:
        parser.error("no wheels given")

    arches = [a.strip() for a in args.arch.split(",") if a.strip()]
    failed = 0
    for wheel in args.wheels:
        if not wheel.is_file():
            print(f"FAIL {wheel}: no such file", file=sys.stderr)
            failed += 1
            continue
        problems = check_wheel(wheel, args.cuda_major, arches)
        if problems:
            failed += 1
            print(f"FAIL {wheel.name}", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
        else:
            print(f"ok   {wheel.name}")
    if failed:
        print(f"\n{failed} of {len(args.wheels)} wheel(s) failed", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
