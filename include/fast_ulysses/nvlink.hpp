#pragma once
/// @file
/// Which pairs of GPUs are joined by NVLink.
///
/// The transport writes peer memory directly, so a group whose GPUs are not NVLink-joined is
/// correct but slower than torch.distributed -- that path routes around the link over a NIC or
/// through host memory, which this one never does. NVML is the only interface that reports the
/// link TYPE; cudaDeviceCanAccessPeer reports only that P2P works, which is equally true of PCIe.
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ulysses {

/// @brief `{(i, j): joined by NVLink}` for every ordered pair of CUDA device indices.
///
/// Two topologies count as joined: a direct GPU-to-GPU link, and both GPUs having an active link
/// into an NVSwitch fabric.
///
/// @return `std::nullopt` when NVML is unavailable or too old to answer. The caller then has no
///         basis to refuse a group and should say so rather than guess.
std::optional<std::map<std::pair<int64_t, int64_t>, bool>> nvlink_matrix(const std::vector<int64_t>& devices);

/// @brief Empty when every pair is NVLink-joined, or when NVML cannot say; otherwise a message
/// naming the first pair that is not.
std::string check_nvlink(const std::vector<int64_t>& devices);

}  // namespace ulysses
