#pragma once
// Binds a result completing on the comm stream to torch's functional-collective machinery, so an
// async call can return an AsyncCollectiveTensor whose .wait() -- or first use by any aten op --
// inserts the cross-stream dependency instead of silently reading a buffer still being written.
// wait_tensor waits every c10d::Work registered against a tensor's STORAGE, and a borrowed result
// is a fresh at::from_blob view per call, so an entry belongs to that CALL, not to the window.
#include <ATen/ATen.h>
#include <cuda_runtime.h>

namespace ulysses {

// Records a completion event on `comm_stream` and binds it to `tensor`; call it with the CALLER's
// stream current. True when the event is now the registry's; false when this build has no registry
// (FAST_ULYSSES_HAS_WORK_REGISTRY=0), in which case nothing could wait on it through torch, so the
// caller's stream is made to wait here and the event destroyed before returning.
bool register_stream_completion(const at::Tensor& tensor, cudaStream_t comm_stream);

}  // namespace ulysses
