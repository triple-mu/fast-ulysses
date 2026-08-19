import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Optional

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).parent.resolve()


def cmake_executable() -> str:
    configured = os.environ.get("FAST_ULYSSES_CMAKE")
    candidates = []
    if configured:
        candidates.append(Path(configured))

    # A pip-installed CMake launcher imports the ``cmake`` Python package. That
    # package is hidden by PEP 517 isolation, but its real binary is still usable.
    if virtual_env := os.environ.get("VIRTUAL_ENV"):
        venv = Path(virtual_env)
        candidates.extend(
            venv.glob("lib/python*/site-packages/cmake/data/bin/cmake")
        )
    if discovered := shutil.which("cmake"):
        candidates.append(Path(discovered))
    candidates.extend([Path("/usr/bin/cmake"), Path("/usr/local/bin/cmake")])

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    raise RuntimeError(
        "CMake was not found. Install it or set FAST_ULYSSES_CMAKE."
    )


def build_jobs() -> int:
    configured = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL") or os.environ.get(
        "FAST_ULYSSES_BUILD_JOBS"
    )
    jobs = int(configured) if configured else min(os.cpu_count() or 4, 32)
    if jobs < 1:
        raise RuntimeError(f"invalid build parallelism: {jobs}")
    return jobs


def ccache_executable() -> Optional[str]:
    configured = os.environ.get("FAST_ULYSSES_CCACHE")
    if configured and configured.lower() in {"0", "false", "off"}:
        return None
    executable = configured or shutil.which("ccache")
    if executable and Path(executable).is_file() and os.access(executable, os.X_OK):
        return str(Path(executable).resolve())
    if configured:
        raise RuntimeError(f"ccache is not executable: {configured!r}")
    return None


def torch_cmake_prefix() -> Path:
    configured = os.environ.get("FAST_ULYSSES_TORCH_PREFIX")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path(sysconfig.get_path("purelib")) / "torch/share/cmake")
    if virtual_env := os.environ.get("VIRTUAL_ENV"):
        candidates.append(
            Path(virtual_env)
            / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
            / "torch/share/cmake"
        )
    for candidate in candidates:
        if (candidate / "Torch/TorchConfig.cmake").is_file():
            return candidate.resolve()
    raise RuntimeError(
        "PyTorch CMake files were not found in the active environment. "
        "Activate the environment that has torch, or set FAST_ULYSSES_TORCH_PREFIX."
    )


def cuda_arch() -> str:
    configured = os.environ.get("FAST_ULYSSES_CUDA_ARCH")
    if configured:
        arch = configured.replace(".", "")
    else:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
        )
        arch = output.splitlines()[0].strip().replace(".", "")
    if not arch.isdigit() or len(arch) < 2:
        raise RuntimeError(f"invalid CUDA architecture: {arch!r}")
    return arch


class CMakeExtension(Extension):
    def __init__(self):
        super().__init__("fast_ulysses._C", sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        output = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        build = ROOT / "build"
        arch = cuda_arch()
        cmake = cmake_executable()
        jobs = build_jobs()
        ccache = ccache_executable()
        env = os.environ.copy()
        env["TORCH_CUDA_ARCH_LIST"] = f"{arch[:-1]}.{arch[-1]}"
        cmake_args = [
            cmake,
            "-S",
            str(ROOT),
            "-B",
            str(build),
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DTORCH_CMAKE_PREFIX_PATH={torch_cmake_prefix()}",
            f"-DEXT_SUFFIX={sysconfig.get_config_var('EXT_SUFFIX') or '.so'}",
            f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        if ccache:
            cmake_args.extend(
                [
                    f"-DCMAKE_CXX_COMPILER_LAUNCHER={ccache}",
                    f"-DCMAKE_CUDA_COMPILER_LAUNCHER={ccache}",
                ]
            )
            env.setdefault("CCACHE_BASEDIR", str(ROOT))
            self.announce(f"using ccache: {ccache}", level=2)
        else:
            cmake_args.extend(
                [
                    "-DCMAKE_CXX_COMPILER_LAUNCHER=",
                    "-DCMAKE_CUDA_COMPILER_LAUNCHER=",
                ]
            )
        self.announce(f"building with {jobs} parallel jobs", level=2)
        subprocess.check_call(
            cmake_args,
            env=env,
        )
        subprocess.check_call(
            [cmake, "--build", str(build), "--parallel", str(jobs)], env=env
        )


setup(
    name="fast-ulysses",
    version="0.3.0.dev0",
    python_requires=">=3.11",
    packages=["fast_ulysses"],
    install_requires=["torch>=2.10"],
    ext_modules=[CMakeExtension()],
    cmdclass={"build_ext": CMakeBuild},
)
