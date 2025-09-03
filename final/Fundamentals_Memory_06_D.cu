/*
Demonstrate a different kind of leak: allocate with `cudaMallocHost` (pinned memory) and forget to free with `cudaFreeHost`.

The aim of this program is to show that pinned (page-locked) host memory allocated with cudaMallocHost is not automatically freed when a CUDA program terminates. By intentionally omitting the cudaFreeHost call, we create a memory leak that persists until the system reclaims the memory (or until the program is killed). This demonstrates a different class of memory leak compared to device memory leaks or host memory leaks. In practice, forgetting to free pinned memory can cause the host process to exhaust its available virtual address space, especially on systems with limited RAM or when repeatedly allocating large buffers in a loop.

To illustrate this, we allocate a small block of pinned memory, perform a trivial operation on it, then exit the program without freeing it. We include basic error checking for the allocation, but deliberately omit the cudaFreeHost call. This should produce a visible leak if you monitor system memory or use CUDA profiling tools after the program terminates.

We do not use any kernel launches or device memory to keep the example focused on host pinned memory. If you run this program on a system with a debugger or memory profiler, you should be able to see that the pinned memory remains allocated after the process exits.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main()
{
    const size_t N = 1024 * 1024; // 1 million floats (~4 MB)
    float *pinnedPtr = nullptr;

    // Allocate pinned (page-locked) memory
    cudaError_t err = cudaMallocHost((void**)&pinnedPtr, N * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Initialize the pinned memory with some values
    for (size_t i = 0; i < N; ++i) {
        pinnedPtr[i] = static_cast<float>(i);
    }

    // Print a small portion to confirm data
    printf("Pinned memory first 5 values: %f %f %f %f %f\n",
           pinnedPtr[0], pinnedPtr[1], pinnedPtr[2], pinnedPtr[3], pinnedPtr[4]);

    // Intentionally omit cudaFreeHost(pinnedPtr) to demonstrate the leak
    // Normally, you would call:
    // cudaFreeHost(pinnedPtr);

    // Optionally reset the CUDA device context
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
