**Synchronous vs. Asynchronous CUDA Errors**

| Type | When It Occurs | How It’s Detected | Typical Examples |
|------|----------------|-------------------|------------------|
| **Synchronous** | Immediately after a CUDA Runtime API call (e.g., `cudaMalloc`, `cudaMemcpy`, `cudaLaunchKernel`, `cudaDeviceSynchronize`). The call itself returns an error code. | The return value of the API function. If it’s not `cudaSuccess`, the error is synchronous. | `cudaMalloc` failing due to insufficient device memory, `cudaMemcpy` failing due to invalid pointer. |
| **Asynchronous** | During or after the execution of a kernel or other work that is launched on the GPU but hasn’t completed yet. Errors are not reported by the launching API call; they surface later when the host checks for errors or when the stream is synchronized. | `cudaGetLastError()` or `cudaPeekAtLastError()` (returns the error from the *most recent* launch in the stream), or a stream synchronization call (`cudaStreamSynchronize`, `cudaDeviceSynchronize`). | Kernel launch failure because of illegal memory access inside the kernel, launching a kernel with an unsupported launch configuration (e.g., too many blocks/threads). |

### How to Identify

1. **Check the return value of the API call** – if it’s not `cudaSuccess`, you have a synchronous error.
2. **After launching a kernel** – call `cudaPeekAtLastError()` immediately; if it returns a non‑`cudaSuccess` value, the kernel launch failed synchronously (e.g., bad launch bounds).
3. **During kernel execution** – the launch may succeed, but the kernel might abort due to an illegal memory access or other fault. This shows up only when the stream is synchronized or when `cudaGetLastError()` is called after the kernel has had a chance to run.

### Quick Example

```c
cudaError_t err;

// Synchronous error example
err = cudaMalloc((void**)&devPtr, size);
if (err != cudaSuccess) {
    // This is a synchronous error
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
}

// Launch kernel
kernel<<<grid, block>>>(devPtr);

// Check for launch errors (synchronous)
err = cudaPeekAtLastError();
if (err != cudaSuccess) {
    // Launch failed synchronously
    fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
}

// Asynchronous error example (e.g., illegal memory access inside kernel)
// Will only surface after synchronization
err = cudaDeviceSynchronize();
if (err != cudaSuccess) {
    // This is an asynchronous error
    fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
}
```

**Bottom line:**  
* Synchronous errors are reported immediately by the API call that triggered them.  
* Asynchronous errors are not reported until you query the device state (e.g., via `cudaDeviceSynchronize()` or `cudaGetLastError()`) after the kernel has run.