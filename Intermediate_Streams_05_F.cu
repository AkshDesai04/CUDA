**Restrictions on a CUDA stream callback (`cudaStreamAddCallback`)**

| What you *can* do | What you *must not* do |
|-------------------|------------------------|
| **Return quickly** – the callback is executed on a host thread that is shared with the CUDA runtime. The function should finish as fast as possible. | **Block** – calling any CUDA runtime API that blocks (e.g., `cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaEventSynchronize`, `cudaDeviceReset`, etc.) will block the CUDA runtime thread and can dead‑lock the host application. |
| **Perform minimal host‑side work** – logging, signaling via atomic variables, setting flags, or updating a simple host data structure. | **Launch kernels** – you should not call `cudaLaunch`, `cudaLaunchKernel`, or any other API that would schedule new work on the GPU. The callback is meant to be called after all work queued in the stream has finished. |
| **Use a tiny set of non‑blocking runtime calls** – the only runtime APIs that are guaranteed to be safe are: <br>`cudaGetLastError()`<br>`cudaPeekAtLastError()`<br>`cudaMemcpyAsync()` (to a *different* stream or to host memory) | **Call most other runtime functions** – for example `cudaMemcpy`, `cudaMemcpyToArray`, `cudaMemcpyFromArray`, `cudaMemcpyToSymbol`, `cudaMemcpyFromSymbol`, `cudaMalloc`, `cudaFree`, `cudaDeviceSetCacheConfig`, `cudaGetDevice`, `cudaSetDevice`, `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaEventCreate`, `cudaEventDestroy`, etc. |
| **Use only host‑side resources** – local stack variables, global or static host data, and atomics/locks that are not CUDA‑specific. | **Modify CUDA context or stream state** – calling `cudaStreamDestroy`, `cudaStreamCreate`, or changing the current device is unsafe from within a callback. |
| **Avoid side effects that might race with other callbacks** – because multiple callbacks can be pending for the same stream, the function must be re‑entrant and thread‑safe. | **Allocate large amounts of memory** – large host allocations can cause the callback to run long, and may interfere with the CUDA runtime’s thread scheduling. |
| **Return a value of type `void`** – the signature is `void callback(cudaStream_t stream, cudaError_t status, void *userData)`. | **Return a non‑void value** – any other return type will lead to undefined behavior. |

### Why these restrictions exist
1. **Thread sharing** – CUDA runtime threads are shared between the host application and the runtime. A blocking call from within the callback would block the runtime thread, preventing the GPU from making progress.
2. **Context ownership** – the callback runs in the context of the host thread that originally created the stream. Changing the context or launching new work can lead to race conditions or deadlocks.
3. **Reentrancy** – multiple callbacks may be queued for the same stream. The callback must be re‑entrant and avoid using static or global state that can be modified concurrently.

### Recommended pattern for a callback
```cpp
__host__ void __stdcall streamCallback(cudaStream_t stream,
                                       cudaError_t status,
                                       void *userData)
{
    // Only read the status, not touch the GPU
    if (status != cudaSuccess) {
        // handle error (e.g., set a flag, log, etc.)
    }

    // Signal host code that stream work is complete
    std::atomic<bool>* done = static_cast<std::atomic<bool>*>(userData);
    done->store(true, std::memory_order_release);

    // No CUDA runtime API calls, no blocking, no kernel launches
}
```

By strictly following these rules, the callback will be safe, efficient, and free from deadlocks or undefined behavior.