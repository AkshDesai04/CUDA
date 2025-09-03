/*
Aim: Use `cudaMemGetInfo` to find out how much free memory is available, then try to allocate slightly more than that amount to trigger the error.

Thought Process:
- The program will first query the CUDA runtime for the amount of free and total device memory using `cudaMemGetInfo`. This API returns the free memory in bytes.
- To guarantee a failure, we will attempt to allocate a buffer that is slightly larger than the reported free memory. Adding a small offset (e.g., 10 MiB) ensures we exceed the limit.
- We'll use `cudaMalloc` to request the allocation and immediately check the returned error code. When the allocation fails, the error should be `cudaErrorMemoryAllocation`, and we will print the corresponding error message.
- For clarity and debugging, the program prints the free/total memory values, the requested allocation size, and the result of the allocation attempt.
- The code is written in standard C++ with CUDA runtime API calls, and it compiles to a `.cu` file. No external dependencies beyond the CUDA toolkit are required.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main()
{
    size_t freeMem, totalMem;
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Device memory: Total = %zu MB, Free = %zu MB\n",
           totalMem / (1024 * 1024),
           freeMem / (1024 * 1024));

    // Try to allocate slightly more than the free memory to trigger an error
    const size_t extraBytes = 10 * 1024 * 1024; // 10 MiB
    size_t allocSize = freeMem + extraBytes;

    printf("Attempting to allocate %zu MB (Free + %zu MB)...\n",
           allocSize / (1024 * 1024),
           extraBytes / (1024 * 1024));

    void* d_ptr = nullptr;
    err = cudaMalloc(&d_ptr, allocSize);

    if (err == cudaSuccess)
    {
        // This should not happen; free the memory and report unexpected success
        printf("Unexpectedly succeeded in allocating %zu MB.\n", allocSize / (1024 * 1024));
        cudaFree(d_ptr);
        return EXIT_SUCCESS;
    }
    else
    {
        printf("Allocation failed as expected: %s\n", cudaGetErrorString(err));
        // Verify that the error is indeed a memory allocation error
        if (err == cudaErrorMemoryAllocation)
            printf("Error code is cudaErrorMemoryAllocation.\n");
        else
            printf("Unexpected error code.\n");
        return EXIT_FAILURE;
    }
}
