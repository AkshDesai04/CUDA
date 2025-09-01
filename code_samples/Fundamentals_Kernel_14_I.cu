A CUDA API call can fail in two fundamentally different ways: **synchronously** or **asynchronously**.

| Aspect | Synchronous error | Asynchronous error |
|--------|-------------------|--------------------|
| **When it occurs** | At the time the API function returns. The host thread sees the error immediately. | After the function returns; the error surfaces only when the device is next accessed from the host (e.g., a later `cudaMemcpy` or `cudaDeviceSynchronize`). |
| **Typical source** | Errors that can be determined on the host side: memory allocation failure (`cudaMalloc`), invalid arguments (`cudaMemcpy` with wrong pointer, wrong size), context‑creation errors, etc. | Errors that require the device to execute work: kernel launch failures due to bad grid/block dimensions, illegal memory accesses inside a kernel, or other runtime errors that manifest only during execution. |
| **Error codes** | Returned directly by the API call (`cudaError_t`). | The call returns `cudaSuccess`, but the device state is marked as *failed*. The next API call that checks the device state will return a non‑zero error. |
| **Recovery** | The error can be handled immediately; you can free resources, retry, or exit. | You must call a *synchronizing* function (e.g., `cudaDeviceSynchronize`, `cudaMemcpy`, or `cudaGetLastError`) to detect the error. Once detected, the context may be in an unrecoverable state, so you often need to reset the device or terminate. |
| **Examples** | `cudaMalloc(&ptr, size)` fails if `size` is too large or the device has no free memory. | Launching a kernel with `<<<blocks, threads>>>()` where `threads > 1024` or `blocks > maxBlocks` may still return `cudaSuccess` immediately, but the launch will fail internally. The error is only seen on a subsequent sync or query. |
| **Timing** | Immediate (host side). | Delayed; depends on the device’s completion of the queued work. |

### How to detect each

* **Synchronous**  
  ```c
  cudaError_t err = cudaMalloc(&ptr, size);
  if (err != cudaSuccess) {
      // handle error right away
  }
  ```

* **Asynchronous**  
  ```c
  kernel<<<blocks, threads>>>(...);   // may launch "silently"
  cudaError_t err = cudaGetLastError();   // checks the *last* launch error
  if (err != cudaSuccess) { ... }

  // or
  err = cudaDeviceSynchronize();   // waits for all work to finish
  if (err != cudaSuccess) { ... }
  ```

In short, synchronous errors are caught immediately on the host because the failure can be detected without waiting for the GPU. Asynchronous errors are deferred until the device has an opportunity to process the request, so you need to explicitly synchronize or query the device to surface them.