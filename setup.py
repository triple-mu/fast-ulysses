from __future__ import annotations

import os
import re
import shlex
import shutil
import site
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

_HERE = Path(__file__).resolve().parent
os.chdir(_HERE)

# Relative to build_lib (and to the wheel root); the source tree keeps the package under python/.
_BUILD_META = Path("fast_ulysses") / "_build_meta.py"
_PKG_DIR = _HERE / "python" / "fast_ulysses"


# Minimum NVSHMEM this extension is built against. 3.4.5 is what the nvidia-nvshmem-cu13
# wheel ships and what every API here was checked against; nothing below it has been tried.
_NVSHMEM_MIN = (3, 4, 5)


def _base_version() -> str:
    """The one version literal, read out of ./VERSION.

    CMake reads the same file, and setup.py bakes it into _build_meta.py, which is where the
    installed package reads __version__ from.
    """
    return (_HERE / "VERSION").read_text().strip()


def _build_jobs(arch: str) -> int:
    """How many translation units to compile at once, bounded by MEMORY, not just cores.

    Each job forks one nvcc, and -t0 makes that nvcc run one device pass per architecture at the
    same time, so peak concurrency is jobs * len(arch) and peak memory scales with it. A CI runner
    with 4 cores and 16 GB is killed by the OOM reaper partway through a four-architecture build if
    -j follows the core count alone. Measured on exactly that shape: eight concurrent passes (-j2,
    four architectures) dies, four (-j1) does not, so a pass costs between 2 and 4 GiB. Budget 3.
    Falls back to the core count where MemAvailable is not readable; CMAKE_BUILD_PARALLEL_LEVEL
    overrides either way.
    """
    cpus = os.cpu_count() or 1
    n_arch = max(1, len([a for a in arch.split(";") if a]))
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                gib = int(line.split()[1]) / (1024 * 1024)
                break
        else:
            return cpus
    except OSError:
        return cpus
    return max(1, min(cpus, int(gib / (3.0 * n_arch))))


def _version() -> str:
    local = os.environ.get("FAST_ULYSSES_LOCAL_VERSION", "")
    return f"{_base_version()}+{local}" if local else _base_version()


def _requires() -> list[str]:
    """Runtime pins derived from the torch this was built against.

    The pin is exact to the torch MINOR: the .so embeds c10d::Work vtables, c10::intrusive_ptr
    in its op schemas, a pybind11 module and a TORCH_LIBRARY registration, and none of those
    survive a minor bump. The nvshmem pin is what guarantees the directory the RPATH points at
    exists.

    Reading it needs the torch this will link against, so the build cannot run under pip's
    build isolation -- and torch cannot be declared a build requirement either, because pip
    would then install its own copy into the isolated environment and the extension would link
    that one instead of the user's. Turn the bare ModuleNotFoundError into the instruction.
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            "fast-ulysses builds against the torch already installed in the target environment, "
            "so it cannot be built under pip's build isolation.\n"
            "    pip install torch                      # the version you intend to run\n"
            "    pip install fast-ulysses --no-build-isolation\n"
            "Prebuilt wheels avoid the source build entirely: see docs/INSTALL.md."
        ) from None

    major, minor = torch.__version__.split(".")[:2]
    cuda_major = torch.version.cuda.split(".")[0]
    return [
        f"torch=={major}.{minor}.*",
        f"nvidia-nvshmem-cu{cuda_major}>={'.'.join(map(str, _NVSHMEM_MIN))}",
    ]


def _nvshmem_version(root: Path) -> tuple[int, int, int] | None:
    """(major, minor, patch) from the install's version header, or None if unreadable."""
    header = root / "include" / "non_abi" / "nvshmem_version.h"
    try:
        text = header.read_text()
    except OSError:
        return None
    parts = []
    for field in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(rf"NVSHMEM_VENDOR_{field}_VERSION\s+(\d+)", text)
        if m is None:
            return None
        parts.append(int(m.group(1)))
    return tuple(parts)  # type: ignore[return-value]


def _nvshmem_candidates() -> list[tuple[str, Path]]:
    """(why, root) in preference order: explicit override, then torch's own wheel, then /usr.

    torch depends on nvidia-nvshmem-cu13, so on a machine that can run this extension the
    library is already installed and version-matched to torch. Using it means one fewer
    thing to install and no chance of loading a second NVSHMEM alongside torch's.
    """
    out: list[tuple[str, Path]] = []
    env = os.environ.get("NVSHMEM_HOME", "")
    if env:
        out.append(("NVSHMEM_HOME", Path(env)))
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        out.append(("torch's nvidia-nvshmem wheel", Path(site_dir) / "nvidia" / "nvshmem"))
    out.append(("system", Path("/usr")))
    return out


def _resolve_nvshmem() -> tuple[Path, Path]:
    """(root, host library). Reports every candidate and why it was rejected."""
    rejected = []
    for why, root in _nvshmem_candidates():
        if not (root / "include" / "nvshmem.h").is_file():
            rejected.append(f"  {why}: {root} -- no include/nvshmem.h")
            continue
        libs = sorted((root / "lib").glob("libnvshmem_host.so*")) or sorted(
            (root / "lib64").glob("libnvshmem_host.so*")
        )
        if not libs:
            rejected.append(f"  {why}: {root} -- no lib/libnvshmem_host.so*")
            continue
        version = _nvshmem_version(root)
        if version is None:
            rejected.append(f"  {why}: {root} -- cannot read include/non_abi/nvshmem_version.h")
            continue
        if version < _NVSHMEM_MIN:
            rejected.append(
                f"  {why}: {root} -- NVSHMEM {'.'.join(map(str, version))}, "
                f"need >= {'.'.join(map(str, _NVSHMEM_MIN))}"
            )
            continue
        print(f"-- NVSHMEM {'.'.join(map(str, version))} from {why}: {root}")
        return root, libs[-1]
    raise RuntimeError(
        "No usable NVSHMEM found. fast_ulysses needs NVSHMEM >= "
        f"{'.'.join(map(str, _NVSHMEM_MIN))} with include/nvshmem.h and "
        "lib/libnvshmem_host.so*. Candidates tried:\n" + "\n".join(rejected) + "\n"
        "Set NVSHMEM_HOME=<install root> to point at one explicitly."
    )


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        nvshmem_home, nvshmem_lib = _resolve_nvshmem()
        outdir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        # Persistent build dir (not pip's ephemeral build_temp) so CMake can
        # build incrementally and skip recompiling unchanged TUs. CMake re-runs
        # configure automatically when the -D args change, so no manual wipe.
        # Overridable so a matrix build gives every config its own tree instead
        # of sharing one with setuptools' own build/lib.* output.
        builddir = Path(os.environ.get("FAST_ULYSSES_BUILD_DIR", _HERE / "build"))
        builddir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        arch = env.get("FAST_ULYSSES_CUDA_ARCH", "80;90;100;120")
        # Torch expects dotted arch (e.g. 86 -> 8.6, 100 -> 10.0); insert the
        # decimal point before the last digit of each ";"-separated token.
        env["TORCH_CUDA_ARCH_LIST"] = " ".join(
            f"{tok[:-1]}.{tok[-1]}" for tok in arch.split(";") if tok
        )
        relocatable = "ON" if env.get("FAST_ULYSSES_RELOCATABLE") == "1" else "OFF"
        # Captured, not streamed: the c10d::register_work probe reports its answer as a status
        # line here and nowhere else outside the compiled extension. Printed either way, so a
        # configure failure still shows what CMake said.
        configure = subprocess.run(
            [
                "cmake",
                "-S",
                str(_HERE),
                "-B",
                str(builddir),
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={outdir}",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-DEXT_SUFFIX={ext_suffix}",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
                f"-DNVSHMEM_HOME={nvshmem_home}",
                f"-DNVSHMEM_HOST_LIB={nvshmem_lib}",
                f"-DFAST_ULYSSES_VERSION={_version()}",
                f"-DFAST_ULYSSES_RELOCATABLE={relocatable}",
            ]
            # Extra -D flags for odd setups (e.g. the CCCL::CCCL stub in docs/INSTALL.md).
            + shlex.split(env.get("FAST_ULYSSES_CMAKE_ARGS", "")),
            env=env,
            stdout=subprocess.PIPE,
            text=True,
        )
        print(configure.stdout, end="", flush=True)
        if configure.returncode != 0:
            raise subprocess.CalledProcessError(configure.returncode, configure.args)
        jobs = env.get("CMAKE_BUILD_PARALLEL_LEVEL") or str(_build_jobs(arch))
        subprocess.check_call(["cmake", "--build", str(builddir), f"-j{jobs}"], env=env)
        self._write_build_meta(
            outdir,
            arch=arch,
            nvshmem_home=nvshmem_home,
            has_work_registry="c10d::register_work available: ON" in configure.stdout,
        )

    def copy_extensions_to_source(self) -> None:
        """--inplace and editable installs copy the extension out of build/lib.* into the
        source tree after the build; the metadata module has to follow it."""
        super().copy_extensions_to_source()
        shutil.copyfile(Path(self.build_lib) / _BUILD_META, _PKG_DIR / _BUILD_META.name)

    @staticmethod
    def _write_build_meta(outdir: Path, *, arch: str, nvshmem_home: Path, has_work_registry: bool):
        """Emit the build facts as a Python module next to the extension.

        build_info() cannot answer them when the dlopen itself failed, which is exactly when
        they are needed. Lands in build/lib.*/fast_ulysses, where bdist_wheel picks it up.
        """
        import torch

        version = _nvshmem_version(nvshmem_home)
        meta = {
            "VERSION": _version(),
            "WHEEL_TAG": os.environ.get("FAST_ULYSSES_LOCAL_VERSION", ""),
            "TORCH_VERSION": torch.__version__,
            "TORCH_ABI": ".".join(torch.__version__.split(".")[:2]),
            "CUDA_VERSION": torch.version.cuda,
            "PYTHON_ABI": f"cp{sys.version_info.major}{sys.version_info.minor}",
            "CUDA_ARCH_LIST": arch.replace(";", ","),
            "NVSHMEM_VERSION": ".".join(map(str, version)) if version else "unknown",
            "HAS_WORK_REGISTRY": has_work_registry,
        }
        (outdir / _BUILD_META.name).write_text(
            "# Generated by setup.py at build time; not in git.\n"
            + "".join(f"{key} = {value!r}\n" for key, value in meta.items())
        )


setup(
    version=_version(),
    install_requires=_requires(),
    ext_modules=[CMakeExtension("fast_ulysses._C")],
    cmdclass={"build_ext": CMakeBuild},
)
