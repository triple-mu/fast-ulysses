"""Name the cause of a failed ``_C`` load.

``ld.so`` reports a mangled symbol or a missing soname, neither of which says WHICH of the
things this extension is pinned to -- the torch minor, the CUDA major, the NVSHMEM wheel --
is not what it was built against. This matches the loader's message against the three known
failures and checks the baked build facts against what is installed now. Imported only from
the failure path, so a working load pays nothing for it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Where the first RPATH entry, $ORIGIN/../nvidia/nvshmem/lib, actually points.
_NVSHMEM_LIB = Path(__file__).resolve().parent.parent / "nvidia/nvshmem/lib/libnvshmem_host.so.3"


def build_meta() -> dict[str, object]:
    """The facts setup.py baked in, or {} when this tree was never built."""
    try:
        from . import _build_meta
    except ImportError:
        return {}
    return {k: v for k, v in vars(_build_meta).items() if k.isupper()}


def _live() -> dict[str, str]:
    try:
        import torch

        version, cuda = torch.__version__, str(torch.version.cuda)
    except Exception:  # noqa: BLE001 -- a torch that will not import is itself a candidate cause
        version, cuda = "unimportable", "unknown"
    return {
        "torch": version,
        "torch_abi": ".".join(version.split(".")[:2]),
        "cuda": cuda,
        "python": f"cp{sys.version_info.major}{sys.version_info.minor}",
    }


def _verdict(text: str, built: dict[str, object], live: dict[str, str]) -> str:
    if not built:
        return "unknown build: _build_meta.py is missing, so there is nothing to compare against."
    if "undefined symbol" in text:
        if built["TORCH_ABI"] != live["torch_abi"]:
            return (
                f"built against torch {built['TORCH_ABI']}, but torch {live['torch_abi']} is "
                "installed, and a torch minor bump changes the C++ ABI this extension links "
                "against. Install a build made for this torch."
            )
        return (
            f"torch {live['torch_abi']} is what this was built against, so the missing symbol "
            "is not a torch minor mismatch."
        )
    if "libcudart" in text:
        if str(built["CUDA_VERSION"]).split(".")[0] != live["cuda"].split(".")[0]:
            return (
                f"built for CUDA {built['CUDA_VERSION']}, but torch reports CUDA {live['cuda']}. "
                "Install the build whose CUDA major matches this torch."
            )
        return "the CUDA major matches the build, so cudart is installed where nothing finds it."
    if "libnvshmem" in text:
        if not _NVSHMEM_LIB.exists():
            return (
                f"{_NVSHMEM_LIB} does not exist. Install nvidia-nvshmem-"
                f"cu{live['cuda'].split('.')[0]}, or put a site NVSHMEM on LD_LIBRARY_PATH."
            )
        return f"{_NVSHMEM_LIB} exists but did not load; check LD_LIBRARY_PATH for another one."
    return "no known failure pattern matched; the loader message below is the whole story."


def explain(exc: BaseException) -> str:
    """A message that leads with the cause and ends with the loader's own words."""
    built, live = build_meta(), _live()
    baked = ", ".join(f"{k.lower()}={v}" for k, v in built.items()) or "unknown build"
    return (
        f"fast_ulysses._C failed to load: {_verdict(str(exc), built, live)}\n"
        f"  built against: {baked}\n"
        f"  installed now: torch={live['torch']}, cuda={live['cuda']}, python={live['python']}, "
        f"nvshmem={_NVSHMEM_LIB if _NVSHMEM_LIB.exists() else 'MISSING'}\n"
        f"  loader error: {exc}"
    )
