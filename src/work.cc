#include <fast_ulysses/work.hpp>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#if FAST_ULYSSES_HAS_WORK_REGISTRY
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#endif

#include <fast_ulysses/common.hpp>

namespace ulysses {

#if FAST_ULYSSES_HAS_WORK_REGISTRY

namespace {

// A Work whose completion is a CUDA event rather than a communicator handle. wait() enqueues a
// dependency on whatever stream is current when the wait happens and returns immediately -- the
// host is not part of the handshake.
class StreamEventWork: public c10d::Work {
public:
    explicit StreamEventWork(cudaEvent_t event): event_(event) {}

    ~StreamEventWork() override
    {
        // Unchecked: this runs from the registry's teardown, where a throw has nowhere to go.
        cudaEventDestroy(event_);
    }

    bool wait(std::chrono::milliseconds /*timeout*/) override
    {
        ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(at::cuda::getCurrentCUDAStream(), event_, 0));
        return true;
    }

    // For a caller that wants the HOST to block, e.g. at teardown.
    void synchronize() override
    {
        ULYSSES_CUDA_CHECK(cudaEventSynchronize(event_));
    }

    bool isCompleted() override
    {
        return cudaEventQuery(event_) == cudaSuccess;
    }

    bool isSuccess() const override
    {
        return true;
    }

private:
    cudaEvent_t event_;
};

}  // namespace

#endif

bool register_stream_completion(const at::Tensor& tensor, cudaStream_t comm_stream)
{
    const at::cuda::CUDAGuard guard(tensor.device());
    cudaEvent_t               event = nullptr;
    ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    ULYSSES_CUDA_CHECK(cudaEventRecord(event, comm_stream));

#if FAST_ULYSSES_HAS_WORK_REGISTRY
    // The registry owns the Work, and the Work owns the event.
    c10d::register_work(tensor, c10::make_intrusive<StreamEventWork>(event));
    return true;
#else
    // Nothing can reach this event through torch, so order it against the caller here. Destroying it
    // immediately is safe: the wait captured the event's recorded state, and destroy defers.
    ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(at::cuda::getCurrentCUDAStream(), event, 0));
    ULYSSES_CUDA_CHECK(cudaEventDestroy(event));
    return false;
#endif
}

}  // namespace ulysses
