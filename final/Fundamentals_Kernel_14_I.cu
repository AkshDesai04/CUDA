In CUDA there are two distinct classes of errors based on when they are detected and reported: **synchronous** and **asynchronous** errors.

| Type | When it is detected | Typical cause | How it is reported | Typical handling |
|------|--------------------|---------------|--------------------|------------------|
| **Synchronous** | **Immediately** during the API call that triggered it | e.g., `cudaMalloc` failing due to insufficient device memory, `cudaMemcpy` with mismatched sizes, invalid context | The function returns a non‑`cudaSuccess` error code right away | Check the return value after the call; if it is an error, handle it (log, free resources, etc.) |
| **Asynchronous** | **Deferred**, after the work has reached the GPU and is executed on the stream | e.g., launching a kernel with an invalid grid/block configuration, using an uninitialized device pointer, out‑of‑bounds accesses inside a kernel | The API call (e.g., kernel launch) usually returns `cudaSuccess`, but a later call such as `cudaGetLastError()` or a synchronization point (e.g., `cudaDeviceSynchronize()`) will reveal the error | After launching, either call `cudaGetLastError()` immediately to catch launch‑time errors, or synchronize and then check the error; any error will be returned by the synchronization call |

### Why the difference matters

- **Error detection timing**  
  - **Synchronous** errors are detected on the host side before any data is transferred to the device.  
  - **Asynchronous** errors are detected on the device side; the host call returns before the kernel actually runs, so the error is only known when the device completes the pending work.

- **Error propagation**  
  - For synchronous errors, the offending API call already indicates failure, so the host can skip subsequent dependent calls.  
  - For asynchronous errors, the kernel launch may have been queued; subsequent launches or copies may still execute, but the error will only surface when you query it or block on the stream.

- **Error handling strategy**  
  - **Synchronous**: check return codes immediately, bail out early.  
  - **Asynchronous**: check `cudaGetLastError()` after kernel launches or before device synchronization; also handle errors that surface during `cudaDeviceSynchronize()` or after other stream‑dependent operations.

### Example

```c
// Synchronous error example
int *dA;
cudaError_t err = cudaMalloc((void**)&dA, 1024 * sizeof(int));
if (err != cudaSuccess) {          // error is caught here
    printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
    return;
}

// Asynchronous error example
myKernel<<<grid, block>>>(dA);     // kernel launch may succeed in the API call
cudaError_t launchErr = cudaGetLastError();  // check if launch itself failed
if (launchErr != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(launchErr));
    return;
}

// Later, when the kernel finishes:
err = cudaDeviceSynchronize();     // catches errors that occur during kernel execution
if (err != cudaSuccess) {
    printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
}
```

**Key takeaway**:  
- **Synchronous errors** are immediate and host‑side.  
- **Asynchronous errors** are delayed, device‑side, and only become visible upon querying or synchronizing. Understanding this distinction is crucial for robust CUDA error handling.