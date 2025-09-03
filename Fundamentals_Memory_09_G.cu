Yes – to obtain a truly *asynchronous* `cudaMemcpyAsync` that can overlap with kernel execution, the host memory involved in the copy must be **page‑locked (pinned)**.

### Why pinned memory is essential

| Aspect | Pageable (normal) memory | Pinned (page‑locked) memory |
|--------|--------------------------|-----------------------------|
| **DMA capability** | The GPU cannot DMA directly into pageable memory; the CUDA runtime must first copy the data from pageable to a temporary pinned buffer. | The GPU can DMA directly from the host buffer because it is guaranteed to stay resident in physical memory. |
| **Transfer mode** | The runtime falls back to a **synchronous** copy path: it waits for the copy to finish (including the extra host‑to‑pinned buffer copy) before returning to the caller. | The copy can be issued immediately to the DMA engine and the call returns without blocking, allowing the CPU to continue. |
| **Overlap with kernels** | Because the copy is synchronous, it blocks the stream until completion; no overlap with kernels in the same stream or other streams is possible. | The copy runs in the background on the DMA engine. If the kernel is launched in a **different** stream, the two operations can execute concurrently on the GPU, giving true overlap. |
| **Performance** | Extra round‑trip copy and possible page faults increase latency and reduce throughput. | Eliminates the extra copy, reduces latency, and maximizes bandwidth. |

### Practical implication

To achieve overlap:

1. **Allocate host memory with `cudaMallocHost` or `cudaHostAlloc`** (or use `cudaHostRegister` on existing page‑locked memory).
2. **Use separate CUDA streams** for the asynchronous copy and the kernel launch.
3. **Ensure that the device memory is allocated with `cudaMalloc`** (device memory is always accessible by DMA).

If any of the host buffers involved in the `cudaMemcpyAsync` are pageable, CUDA internally creates a temporary pinned buffer, performs the copy, and blocks the calling stream until the transfer completes. Consequently, the copy cannot truly overlap with kernel execution, and the benefit of `cudaMemcpyAsync` is lost.

In short, **pinned memory is mandatory for asynchronous, overlapping host‑to‑device or device‑to‑host transfers** in CUDA.