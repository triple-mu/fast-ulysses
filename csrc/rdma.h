#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ulysses {

// The supported 8-GPU host splits into two quads. Peers inside a quad are reachable through CUDA
// IPC pointers and are copied to directly; peers across quads are reached only through the NIC.
// Every peer index arithmetic in the transport rests on this split.
constexpr int kQuad = 4;

// An interleaved MKey's stride is a 16-bit field. A larger one is accepted by every verbs call
// on the way in and then gathers the wrong bytes at transfer time -- no completion reports it --
// so a shape that needs one has to be refused up front.
constexpr int64_t kMaxInterleavedStride = 65535;

class RdmaBuffer {
public:
    ~RdmaBuffer();

private:
    friend class RdmaTransport;
    struct Impl;
    explicit RdmaBuffer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class RdmaTransport {
public:
    // nics is either empty, for sysfs discovery, or one mlx5 name per rank. Both it and enable
    // come from the environment, which is read in Python.
    RdmaTransport(int rank, int world_size, int device,
                  const std::vector<int64_t>& devices,
                  bool enable, const std::vector<std::string>& nics);
    ~RdmaTransport();

    bool enabled() const;
    // Why this shape cannot go over the NIC, or "" if it can. Answered by dry-running the
    // geometry the transfer itself will program, not by a second copy of the rules: a shape
    // that passes here and then fails inside register_buffer would make the query a lie.
    std::string shape_reason(int mode,
                             int64_t batch,
                             int64_t seq,
                             int64_t heads,
                             int64_t dim,
                             int64_t element_size) const;
    std::vector<int64_t> connection_info() const;
    void connect(const std::vector<std::vector<int64_t>>& peers);

    std::unique_ptr<RdmaBuffer> register_buffer(void* pointer,
                                                int64_t bytes,
                                                int mode,
                                                int64_t batch,
                                                int64_t seq,
                                                int64_t heads,
                                                int64_t dim,
                                                int64_t element_size);
    std::vector<int64_t> buffer_info(const RdmaBuffer& buffer) const;
    void connect_buffer(RdmaBuffer& buffer,
                        const std::vector<std::vector<int64_t>>& peers) const;
    std::vector<uint64_t> peer_pointers(const RdmaBuffer& buffer) const;
    std::vector<uint64_t> peer_flags(const RdmaBuffer& buffer) const;
    // Close only mappings imported from peers. The caller coordinates this before any rank
    // releases its locally exported output/flag allocations. Safe to retry after a partial
    // close and safe to call again after success.
    void close_buffer_imports(RdmaBuffer& buffer) const;
    void start_exchange(const void* input,
                        int64_t input_bytes,
                        RdmaBuffer& output,
                        int mode,
                        int64_t batch,
                        int64_t seq,
                        int64_t heads,
                        int64_t dim,
                        int64_t element_size);
    void finish_exchange();
    void flush() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ulysses
