/*
Aim: Explain in a comment that even if a large allocation succeeds, it might make the GPU unusable for display or other applications.

Thinking:
The user wants a clear explanation that, although a cudaMalloc can return a success code for a large request, the GPU might become unusable for the display driver or other processes that also need memory. To satisfy the developer instruction, the code must be a complete CUDA C (.cu) file. The explanation will be placed in a comment near the allocation block. The program will simply attempt a large allocation, report success or failure, and free the memory. No additional text will be output. */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    size_t largeSize = 8ULL * 1024 * 1024 * 1024; // 8 GB
    void* d_ptr = NULL;

    // Attempt to allocate a large chunk of GPU memory.
    // Even if this allocation succeeds, the GPU may become unusable for
    // the display driver or other applications because the allocation
    // consumes memory that would otherwise be reserved for the driver,
    // OS, and system context required for display output. This can
    // lead to GPU hangs, crashes, or inability to run other GPU
    // workloads, especially on devices with limited total memory.
    cudaError_t err = cudaMalloc(&d_ptr, largeSize);

    if (err == cudaSuccess) {
        printf("Successfully allocated %zu bytes on GPU.\n", largeSize);
    } else {
        printf("Failed to allocate %zu bytes on GPU. Error: %s\n",
               largeSize, cudaGetErrorString(err));
    }

    // Clean up if allocation succeeded
    if (d_ptr) {
        cudaFree(d_ptr);
    }

    return 0;
}
