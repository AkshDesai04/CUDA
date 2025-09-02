**Synchronous vs. Asynchronous CUDA Errors**

| Type | Where it occurs | How it is reported | When you see it | Typical example |
|------|-----------------|--------------------|-----------------|-----------------|
| **Synchronous error** | Immediately at the CUDA API call that triggers it | Returned as the `cudaError_t` from the function itself | Right after the call, before the program proceeds | `cudaMalloc()` failing because of insufficient device memory |
| **Asynchronous error** | During a later operation that was already launched | Not returned by the launch itself; appears on a subsequent CUDA call that synchronizes with the device | After the device has finished executing a queued work item (e.g., after `cudaDeviceSynchronize()`, `cudaMemcpy()` to/from device, or another kernel launch) | A kernel launch with an invalid configuration (e.g., too many blocks or threads) or accessing out‑of‑bounds memory inside the kernel |

---

### 1. **Synchronous Errors**

- **When they happen**  
  The error is detected by the host code as soon as the API call is made. The host driver performs all necessary validation (argument types, sizes, device availability, etc.) and returns an error code immediately.

- **Typical sources**  
  - Invalid parameters (negative size, bad pointer, wrong stream, etc.)  
  - Resource exhaustion on the host (e.g., `malloc` failing)  
  - Wrong device index or context

- **Error handling**  
  Since the error code is returned directly, you can check it right after the call and handle it (e.g., retry, abort, or clean up). No need to wait for the device.

---

### 2. **Asynchronous Errors**

- **When they happen**  
  These errors are detected on the device side **after** the work has been queued. The CUDA driver launches the kernel or starts a memory copy, which are asynchronous relative to the host. The host code continues executing. The error is not visible until a later host API call forces a synchronization with the device.

- **Typical sources**  
  - Bad launch configuration (e.g., too many blocks, too many threads per block, exceeding resource limits)  
  - Kernel accessing out‑of‑bounds memory or illegal memory addresses  
  - Using an uninitialized pointer or a freed device pointer inside a kernel  
  - Wrong shared memory usage

- **Error handling**  
  Since the error isn’t returned by the launch itself, you need to call a **synchronizing** API to surface it:
  ```cpp
  cudaMemcpy(...);          // or any other sync call
  cudaError_t err = cudaGetLastError(); // or check the error from the sync call
  ```
  After this, `err` will contain the actual error that occurred during the earlier asynchronous operation.

---

### 3. **Practical Implications**

| Scenario | What to do immediately | What to do later |
|----------|------------------------|------------------|
| `cudaMalloc` fails | Check return value; abort or retry | N/A |
| Kernel launch fails (bad config) | Ignore launch return (always `cudaSuccess`) | Call `cudaGetLastError()` or `cudaDeviceSynchronize()` to capture the error |
| Out‑of‑bounds memory access inside kernel | Cannot be detected at launch | Detected on next sync; use `cudaGetLastError()` or `cudaDeviceSynchronize()` |

---

### 4. **Key Takeaway**

- **Synchronous errors** are caught **at the point of the call**; they can be handled immediately.
- **Asynchronous errors** are caught **after the device has executed or attempted the queued work**; they become visible only after a host–device synchronization. Proper error checking in CUDA programs therefore requires a mix of immediate checks and later synchronization checks (`cudaGetLastError()` and/or `cudaDeviceSynchronize()`).
