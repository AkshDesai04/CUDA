```cpp
/*
Demonstrate a `cudaErrorMemoryAllocation` error by attempting to allocate an impossibly large amount of memory on the device (e.g., 100 GB). Catch and report the error gracefully.

To achieve this, we will write a simple CUDA C program that attempts to allocate 100 gigabytes of device memory using cudaMalloc. 100 GB is far beyond the capacity of most GPUs, so we expect cudaMalloc to fail with cudaErrorMemoryAllocation.  The program will check the return value of cudaMalloc, and if an error occurs, it will print a descriptive message including the CUDA error string.  If for some reason the allocation succeeds (e.g., on a system with an extremely large device memory), the program will free the memory and report success.  No additional functionality is required, so the code remains minimal and focused on demonstrating the error handling.

The code includes:
- Standard headers and the CUDA runtime header.
- A main function that calculates the requested allocation size (100 * 1024^3 bytes).
- Calls to cudaMalloc and error checking.
- Graceful error handling with a user-friendly message.
- Cleanup via cudaFree if allocation succeeded.

This program can be compiled with nvcc (e.g., `nvcc -o memalloc_demo memalloc_demo.cu`) and run on a machine with a CUDA-capable GPU.  It will print the appropriate error message to the console, illustrating how to handle memory allocation failures in CUDA.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const size_t GB = 1024ULL * 1024ULL * 1024ULL;
    const size_t requestedGB = 100ULL;
    size_t sizeBytes = requestedGB * GB; // 100 GB

    void *d_ptr = NULL;
    cudaError_t err = cudaMalloc(&d_ptr, sizeBytes);

    if (err != cudaSuccess) {
        fprintf(stderr,
                "cudaMalloc failed for %zu GB (%zu bytes): %s\n",
                requestedGB, sizeBytes, cudaGetErrorString(err));
        // No need to free because allocation failed
        return EXIT_FAILURE;
    } else {
        printf("cudaMalloc succeeded for %zu GB (%zu bytes). This is unexpected on most GPUs.\n",
               requestedGB, sizeBytes);
        // Free the allocated memory
        cudaFree(d_ptr);
        return EXIT_SUCCESS;
    }
}
```