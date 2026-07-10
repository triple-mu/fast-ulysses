from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

_HERE = Path(__file__).resolve().parent
os.chdir(_HERE)

NVSHMEM_HOME = os.environ.get(
    "NVSHMEM_HOME", "/data/libnvshmem-linux-x86_64-3.7.0_cuda12-archive"
)
if not (Path(NVSHMEM_HOME) / "include" / "nvshmem.h").exists():
    raise FileNotFoundError(f"NVSHMEM_HOME invalid: {NVSHMEM_HOME}")


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
        builddir = _HERE / "build"
        builddir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        arch = env.get("CUSTOM_ULYSSES_CUDA_ARCH", "80;90;100;120")
        # Torch expects dotted arch (e.g. 86 -> 8.6, 100 -> 10.0); insert the
        # decimal point before the last digit of each ";"-separated token.
        env["TORCH_CUDA_ARCH_LIST"] = " ".join(
            f"{tok[:-1]}.{tok[-1]}" for tok in arch.split(";") if tok
        )
        subprocess.check_call(
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
                f"-DNVSHMEM_HOME={NVSHMEM_HOME}",
            ],
            env=env,
        )
        subprocess.check_call(["cmake", "--build", str(builddir), "-j"], env=env)


setup(
    name="fast-ulysses",
    version="0.0.1",
    packages=["fast_ulysses"],
    ext_modules=[CMakeExtension("fast_ulysses._C")],
    cmdclass={"build_ext": CMakeBuild},
)
