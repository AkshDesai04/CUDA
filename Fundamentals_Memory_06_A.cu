/*
Write the memory-leaking code inside a function. Call this function in a loop in `main` to leak memory more quickly.

Thinking:
- The goal is to intentionally allocate GPU memory repeatedly without freeing it, creating a memory leak.
- We will create a function `leakMemory()` that performs `cudaMalloc` for a fixed amount of memory (e.g., 1 MB) and does not call `cudaFree`.
- In `main`, we will call `leakMemory()` inside a loop (e.g., 10,000 iterations) to accelerate the leak.
- Error checking will be minimal; if `cudaMalloc` fails we will print an error and return, but still not free any memory.
- This program will compile as a CUDA source file (.cu) and requires linking with the CUDA runtime.
- The code uses standard C includes and the CUDA runtime header.
*/

#include <stdio.h>
#include <cuda_runtime.h>

void leakMemory()
{
    void* devPtr = NULL;
    size_t size = 1024 * 1024; // 1 MB
    cudaError_t err = cudaMalloc(&devPtr, size);
    if (err != cudaSuccess)
    {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return;
    }
    // Intentionally do not free devPtr to create a memory leak
}

int main()
{
    const int numIterations = 10000;
    for (int i = 0; i < numIterations; ++i)
    {
        leakMemory();
        // Optional: small delay to avoid overwhelming the system instantly
        // usleep(1000);
    }
    printf("Completed %d memory leak iterations.\n", numIterations);
    return 0;
}