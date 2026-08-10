from __future__ import annotations

import os
import shlex
import shutil
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


def _base_version() -> str:
    """The one version literal, read out of ./VERSION.

    CMake reads the same file, and setup.py bakes it into _build_meta.py, which is where the
    installed package reads __version__ from.
    """
    return (_HERE / "VERSION").read_text().strip()


def _build_jobs(arch: str) -> int:
    """How many translation units to compile at once, bounded by MEMORY, not just cores.

    Each job forks one nvcc, and -t0 makes that nvcc run one device pass per architecture at the
    same time, so peak concurrency is jobs * len(arch) and peak memory scales with it. A small CI
    runner is killed by the OOM reaper partway through a four-architecture build if -j follows the
    core count alone. Budget 3 GiB per concurrent pass. Falls back to the core count where
    MemAvailable is not readable; CMAKE_BUILD_PARALLEL_LEVEL overrides either way.
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
    """The runtime pin, derived from the torch this was built against.

    torch is the only runtime dependency, and the pin is exact to its MINOR: the .so embeds
    c10d::Work vtables, c10::intrusive_ptr in its op schemas, a pybind11 module and a
    TORCH_LIBRARY registration, and none of those survive a minor bump.

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
            "Prebuilt wheels avoid the source build entirely: see docs/install.md."
        ) from None

    major, minor = torch.__version__.split(".")[:2]
    return [f"torch=={major}.{minor}.*"]


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
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
                f"-DFAST_ULYSSES_VERSION={_version()}",
            ]
            # Extra -D flags for odd toolkit layouts; see docs/install.md.
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
            has_work_registry="c10d::register_work available: ON" in configure.stdout,
        )

    def copy_extensions_to_source(self) -> None:
        """--inplace and editable installs copy the extension out of build/lib.* into the
        source tree after the build; the metadata module has to follow it."""
        super().copy_extensions_to_source()
        shutil.copyfile(Path(self.build_lib) / _BUILD_META, _PKG_DIR / _BUILD_META.name)

    @staticmethod
    def _write_build_meta(outdir: Path, *, arch: str, has_work_registry: bool):
        """Emit the build facts as a Python module next to the extension.

        build_info() cannot answer them when the dlopen itself failed, which is exactly when
        they are needed. Lands in build/lib.*/fast_ulysses, where bdist_wheel picks it up.
        """
        import torch

        meta = {
            "VERSION": _version(),
            "WHEEL_TAG": os.environ.get("FAST_ULYSSES_LOCAL_VERSION", ""),
            "TORCH_VERSION": torch.__version__,
            "TORCH_ABI": ".".join(torch.__version__.split(".")[:2]),
            "CUDA_VERSION": torch.version.cuda,
            "PYTHON_ABI": f"cp{sys.version_info.major}{sys.version_info.minor}",
            "CUDA_ARCH_LIST": arch.replace(";", ","),
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
