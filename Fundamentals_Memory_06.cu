```cuda
/*
Aim of the program: 
Write a program that deliberately leaks memory on the device by allocating it but never calling `cudaFree`. 
Run the program multiple times and use the `nvidia-smi` command-line tool to observe how the GPU's memory usage increases.

Thinking:
- We want a minimal CUDA program that allocates memory on the GPU and does not free it.
- The simplest way to leak memory across program runs is to allocate some amount of device memory in main and exit without freeing it.
- Allocate a reasonable chunk (e.g., 50 MB) to make the effect visible when checking with `nvidia-smi`.
- We keep the pointer in a static array so the compiler cannot optimize away the allocation.
- No error handling is strictly necessary, but a simple check is added for clarity.
- After allocation, we can print the pointer address and maybe perform a trivial kernel launch to keep the context alive, but it is not required.
- The program will exit immediately after allocation, leaving the GPU memory allocated (since we never call `cudaFree`).
- Running this program multiple times will cumulatively increase the GPU memory usage, which can be observed with `nvidia-smi`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    const size_t allocationSize = 50 * 1024 * 1024; // 50 MB
    void *devPtr = NULL;

    cudaError_t err = cudaMalloc(&devPtr, allocationSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Allocated %zu bytes on device at pointer %p. Not freeing.\n", allocationSize, devPtr);

    // Keep the pointer alive in a static array to avoid compiler optimization
    static void *allocations[1];
    allocations[0] = devPtr;

    // We intentionally do NOT call cudaFree(devPtr);

    return 0;
}
```