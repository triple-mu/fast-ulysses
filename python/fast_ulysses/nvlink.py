"""Which pairs of GPUs are joined by NVLink.

The transport writes peer memory directly, so a group whose GPUs are not NVLink-joined is correct
but slower than ``torch.distributed`` -- that path routes around the link over a NIC or through
host memory, which we never do. Refusing such a group at construction is cheaper than letting
someone find it in a profile.

NVML is the only interface that reports the link TYPE. ``cudaDeviceCanAccessPeer`` reports only
that P2P works, which is equally true over PCIe.
"""

from __future__ import annotations

import ctypes
import functools

_NVML_SUCCESS = 0
_MAX_LINKS = 32  # past NVML_NVLINK_MAX_LINKS (18 today); the sweep stops on the first error
_REMOTE_GPU = 0  # NVML_NVLINK_DEVICE_TYPE_GPU
_REMOTE_SWITCH = 2  # NVML_NVLINK_DEVICE_TYPE_SWITCH


class _PciInfo(ctypes.Structure):
    """nvmlPciInfo_t. Only domain/bus/device are read, and only after the link's remote end has
    already said it is a GPU; an unmatched result is reported as "cannot determine" rather than as
    a missing link, so a layout change cannot turn into a wrong refusal."""

    _fields_ = [
        ("busIdLegacy", ctypes.c_char * 16),
        ("domain", ctypes.c_uint),
        ("bus", ctypes.c_uint),
        ("device", ctypes.c_uint),
        ("pciDeviceId", ctypes.c_uint),
        ("pciSubSystemId", ctypes.c_uint),
        ("busId", ctypes.c_char * 32),
    ]


@functools.lru_cache(maxsize=1)
def _nvml():
    """The initialised NVML handle, or None if it is unavailable or too old for what we ask."""
    try:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
    except OSError:
        return None
    if lib.nvmlInit_v2() != _NVML_SUCCESS:
        return None
    for name in (
        "nvmlDeviceGetHandleByPciBusId_v2",
        "nvmlDeviceGetNvLinkState",
        "nvmlDeviceGetNvLinkRemoteDeviceType",
        "nvmlDeviceGetNvLinkRemotePciInfo_v2",
    ):
        if not hasattr(lib, name):
            return None
    return lib


def _handles(lib, devices: list[int]) -> dict[int, ctypes.c_void_p] | None:
    import torch

    out: dict[int, ctypes.c_void_p] = {}
    for i in devices:
        p = torch.cuda.get_device_properties(i)
        bus_id = f"{p.pci_domain_id:08x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0".encode()
        h = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByPciBusId_v2(bus_id, ctypes.byref(h)) != _NVML_SUCCESS:
            return None
        out[i] = h
    return out


def nvlink_matrix(devices: list[int]) -> dict[tuple[int, int], bool] | None:
    """``{(i, j): joined by NVLink}`` for every ordered pair of CUDA device indices.

    Two topologies count as joined: a direct GPU-to-GPU link, and both GPUs having an active link
    into an NVSwitch fabric. ``None`` means NVML could not answer -- the caller then has no basis
    to refuse a group and should say so rather than guess.
    """
    import torch

    lib = _nvml()
    if lib is None:
        return None
    handles = _handles(lib, devices)
    if handles is None:
        return None
    by_bdf = {
        (
            torch.cuda.get_device_properties(i).pci_domain_id,
            torch.cuda.get_device_properties(i).pci_bus_id,
            torch.cuda.get_device_properties(i).pci_device_id,
        ): i
        for i in devices
    }

    linked = {(i, j): i == j for i in devices for j in devices}
    on_switch: set[int] = set()
    answered = False

    for i, h in handles.items():
        for link in range(_MAX_LINKS):
            active = ctypes.c_uint()
            if lib.nvmlDeviceGetNvLinkState(h, link, ctypes.byref(active)) != _NVML_SUCCESS:
                break  # past this device's link count
            answered = True
            if not active.value:
                continue
            kind = ctypes.c_uint()
            if (
                lib.nvmlDeviceGetNvLinkRemoteDeviceType(h, link, ctypes.byref(kind))
                != _NVML_SUCCESS
            ):
                continue
            if kind.value == _REMOTE_SWITCH:
                on_switch.add(i)
            elif kind.value == _REMOTE_GPU:
                info = _PciInfo()
                if (
                    lib.nvmlDeviceGetNvLinkRemotePciInfo_v2(h, link, ctypes.byref(info))
                    != _NVML_SUCCESS
                ):
                    continue
                peer = by_bdf.get((info.domain, info.bus, info.device))
                if peer is not None:
                    linked[(i, peer)] = linked[(peer, i)] = True

    if not answered:
        return None
    # One fabric per node, so every GPU with a switch link can reach every other one.
    for i in on_switch:
        for j in on_switch:
            linked[(i, j)] = True
    return linked


def check_nvlink(devices: list[int]) -> str | None:
    """``None`` when every pair is NVLink-joined, or when NVML cannot say; otherwise a message
    naming the first pair that is not."""
    matrix = nvlink_matrix(devices)
    if matrix is None:
        return None
    for i in devices:
        for j in devices:
            if i != j and not matrix[(i, j)]:
                return (
                    f"cuda:{i} and cuda:{j} are not joined by NVLink. fast-ulysses writes peer "
                    "memory directly, which over PCIe -- and especially across a CPU socket -- is "
                    "slower than torch.distributed, because that routes around the link instead. "
                    "Use torch.distributed for this group, or pass require_nvlink=False to "
                    "measure it anyway. `fast-ulysses doctor` prints the full matrix."
                )
    return None
