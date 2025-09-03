/*
Aim: Demonstrate a `cudaErrorMemoryAllocation` error by attempting to allocate an impossibly large amount of memory on the device (e.g., 100 GB). Catch and report the error gracefully.

Thinking:
- The goal is to trigger a `cudaErrorMemoryAllocation` error in a controlled manner.
- On almost all GPUs, allocating 100 GB will exceed available device memory.
- We'll use `cudaMalloc` to request 100 GB of device memory and check the returned error code.
- If the allocation fails, we'll print a friendly error message that includes the error string.
- If, improbably, the allocation succeeds (e.g., on a system with >100 GB device memory), we'll free the memory to avoid leaks.
- The program will compile and run with minimal dependencies: `<cstdio>`, `<cuda_runtime.h>`, and standard I/O.
- We'll ensure the program exits gracefully regardless of success or failure.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    const size_t gigabytes = 100ull;                // Request 100 GB
    const size_t bytes_per_gb = 1024ull * 1024ull * 1024ull; // 1 GB in bytes
    const size_t alloc_size = gigabytes * bytes_per_gb;     // Total allocation size in bytes

    void* d_ptr = nullptr;
    cudaError_t err = cudaMalloc(&d_ptr, alloc_size);

    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate %zu GB on device. Error: %s (code %d)\n",
                gigabytes, cudaGetErrorString(err), err);
        // No memory to free since allocation failed
        return EXIT_FAILURE;
    }

    // If we reach here, allocation succeeded (unlikely)
    printf("Successfully allocated %zu GB on device. Freeing memory.\n", gigabytes);

    // Free the allocated memory
    err = cudaFree(d_ptr);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to free device memory. Error: %s (code %d)\n",
                cudaGetErrorString(err), err);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
