Common synchronous CUDA API calls include:

1. `cudaMemcpy()` – copies memory between host and device or between device pointers, and blocks until the copy completes.
2. `cudaDeviceSynchronize()` – blocks the host until all previously issued device work (kernels, memory operations, etc.) has finished.
3. `cudaFree()` – deallocates device memory and waits until the deallocation completes before returning.