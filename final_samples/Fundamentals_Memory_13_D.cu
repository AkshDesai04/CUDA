/*
Aim of the program: Write a `try_alloc(size_t bytes)` function that returns `true` if `cudaMalloc` succeeds (and then immediately frees it) and `false` otherwise.

Thinking:
The goal is to create a simple CUDA helper that tests whether a specified amount of GPU memory can be allocated.  
- Include the CUDA runtime header (`cuda_runtime.h`) and standard headers for types and I/O.  
- Use a host‑only function `bool try_alloc(size_t bytes)` that attempts `cudaMalloc`.  
- If `cudaMalloc` returns `cudaSuccess`, free the allocated memory with `cudaFree` and return `true`.  
- Otherwise, return `false`.  
- A small `main` function demonstrates usage by trying a modest allocation (1 MB) and a large allocation (8 GB).  
- The program is written in plain C/C++ so it can be compiled with `nvcc`.  
- `bool` comes from `<stdbool.h>`; `size_t` from `<stddef.h>`.  
- Error strings are printed only for debugging; the core functionality is the return value.  
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>

/**
 * Attempts to allocate `bytes` of device memory.
 * @param bytes Number of bytes to allocate.
 * @return true if allocation succeeded (and was immediately freed), false otherwise.
 */
bool try_alloc(size_t bytes)
{
    void* ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, bytes);

    if (err == cudaSuccess)
    {
        // Allocation succeeded; free the memory and report success.
        cudaFree(ptr);
        return true;
    }

    // Allocation failed; no need to free as cudaMalloc did not succeed.
    return false;
}

int main()
{
    // Test with a small allocation (1 MB).
    size_t small = 1024 * 1024; // 1 MB
    printf("Trying small allocation (%zu bytes): %s\n",
           small, try_alloc(small) ? "succeeded" : "failed");

    // Test with a large allocation (8 GB).
    size_t large = 8ULL * 1024 * 1024 * 1024; // 8 GB
    printf("Trying large allocation (%zu bytes): %s\n",
           large, try_alloc(large) ? "succeeded" : "failed");

    return 0;
}
