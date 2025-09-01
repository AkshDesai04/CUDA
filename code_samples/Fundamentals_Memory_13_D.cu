/*
Aim: Write a `try_alloc(size_t bytes)` function that returns `true` if `cudaMalloc` succeeds (and then immediately frees it) and `false` otherwise.

Thinking:
- The goal is to provide a lightweight wrapper around `cudaMalloc` that checks whether a given amount of GPU memory can be allocated.
- The function should:
  1. Accept a `size_t` argument specifying the number of bytes to allocate.
  2. Attempt to allocate memory on the device using `cudaMalloc`.
  3. If the allocation succeeds, immediately free the memory with `cudaFree` to avoid leaking GPU resources.
  4. Return `true` if the allocation succeeded, `false` otherwise.
- Since the function interacts with the CUDA runtime, we need to include `<cuda_runtime.h>`.
- The return type is `bool`; thus, include `<stdbool.h>` or use C++ style `bool`. The file is a `.cu` CUDA source, which compiles as C++ by default, so `bool` is fine.
- For safety, we will check the return status of `cudaMalloc` and `cudaFree`. If `cudaFree` fails, we will still return `true` because the allocation succeeded; freeing failure is a separate issue, but we will log an error message.
- The code will include a minimal `main` function to demonstrate usage (optional). The main function will read a byte size from the user, call `try_alloc`, and print the result.
- All code is placed in a single `.cu` file.

Note: To compile: `nvcc -o try_alloc try_alloc.cu`
*/

#include <cstdio>
#include <cuda_runtime.h>
#include <cstddef>   // for size_t

/**
 * Attempts to allocate `bytes` bytes of device memory.
 *
 * @param bytes Number of bytes to allocate.
 * @return `true` if allocation succeeded, `false` otherwise.
 */
bool try_alloc(size_t bytes)
{
    void* dev_ptr = nullptr;
    cudaError_t err = cudaMalloc(&dev_ptr, bytes);
    if (err != cudaSuccess) {
        // Allocation failed
        return false;
    }

    // Allocation succeeded; free the memory immediately
    cudaError_t free_err = cudaFree(dev_ptr);
    if (free_err != cudaSuccess) {
        // Free failed; log but still consider allocation successful
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(free_err));
    }

    return true;
}

int main()
{
    size_t bytes = 0;
    printf("Enter number of bytes to test allocation: ");
    if (scanf("%zu", &bytes) != 1) {
        fprintf(stderr, "Invalid input.\n");
        return 1;
    }

    bool success = try_alloc(bytes);
    if (success) {
        printf("Successfully allocated and freed %zu bytes on the GPU.\n", bytes);
    } else {
        printf("Failed to allocate %zu bytes on the GPU.\n", bytes);
    }

    return 0;
}