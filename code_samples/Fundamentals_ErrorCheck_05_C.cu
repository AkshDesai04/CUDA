A CUDA error is considered **synchronous** if it can be detected immediately by the call that generates it.  
Examples of synchronous errors are:

* `cudaMalloc()` failing because of insufficient host memory – the failure is reported back to the host right away.  
* `cudaMemcpy()` with a host‑to‑device copy that violates the bounds of the source or destination – the error is returned as soon as the API call returns.  
* Trying to launch a kernel with an invalid configuration (e.g., zero blocks or threads, or an invalid grid/block dimension) – the runtime checks the parameters before the kernel is scheduled and returns an error synchronously.

A CUDA error is considered **asynchronous** if it is detected only after the kernel has been launched and the error surfaces later, typically when the host code subsequently synchronizes with the device.  
Examples of asynchronous errors are:

* The device running out of memory during a kernel launch – the kernel launch may succeed, but the runtime reports `cudaErrorOutOfMemory` only when the host later calls `cudaDeviceSynchronize()` or another blocking API that forces the device to finish the kernel.  
* A kernel that accesses memory out of bounds – the error is only seen when the host synchronizes.  
* A launch that triggers a launch failure (e.g., due to exceeding the maximum number of concurrent kernels) – the failure is reported asynchronously.

In short:  
**Synchronous error** – detected and reported by the same API call.  
**Asynchronous error** – detected only when a later synchronization or blocking operation forces the device to finish execution.