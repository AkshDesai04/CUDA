/*
Aim: Demonstrate a different kind of leak: allocate with `cudaMallocHost` (pinned memory) and forget to free with `cudaFreeHost`.

Thinking:
- The goal is to create a minimal CUDA C/C++ program that intentionally leaks pinned memory by allocating with `cudaMallocHost` and not freeing it with `cudaFreeHost`.
- We'll allocate a modest amount of pinned memory (e.g., 1 MB) to keep the example simple and observable.
- The program will perform a trivial operation (like filling the memory with zeros) to illustrate that the allocation can be used.
- We will include basic error checking for the CUDA API calls.
- No deallocation of the pinned memory is performed, so the leak remains until the process exits.
- `cudaDeviceReset()` is called before program termination to ensure any device-side resources are cleaned up, but the host pinned allocation remains allocated.
- The code is written in a single .cu file with a clear `main` function and necessary includes.

*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main(void)
{
    const size_t size = 1 * 1024 * 1024; // 1 MB
    float *pinnedMem = nullptr;

    // Allocate pinned (page-locked) host memory
    cudaError_t err = cudaMallocHost((void **)&pinnedMem, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Use the memory: initialize to zero
    for (size_t i = 0; i < size / sizeof(float); ++i) {
        pinnedMem[i] = 0.0f;
    }

    printf("Pinned memory allocated and initialized. Size: %zu bytes.\n", size);
    printf("This program intentionally does NOT free the pinned memory to demonstrate a leak.\n");

    // Normally we would free with cudaFreeHost(pinnedMem);
    // Here we omit that call intentionally.

    // Reset the device (good practice)
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
