/*
Write a 'Hello World' CUDA program. The kernel should be launched with a single block containing a single thread. The host code (the `main` function) should print 'Kernel launched successfully!' after the kernel launch. The kernel itself will be empty.

The goal is to provide a minimal, compilable CUDA C program that demonstrates launching a kernel with one block and one thread, even though the kernel does nothing. After the launch, the host prints a confirmation message. The program should be self-contained and compile with a typical CUDA compiler (`nvcc`). 

Key steps:
1. Include necessary headers (`<stdio.h>`, `<cuda_runtime.h>`).
2. Define an empty global kernel (`__global__ void helloWorldKernel()`).
3. In `main`, launch the kernel with `<<<1, 1>>>`.
4. Optionally, check for CUDA launch errors with `cudaGetLastError()` and `cudaDeviceSynchronize()` for safety.
5. Print "Kernel launched successfully!" after the launch.
6. Return from `main`.

No additional functionality or error handling beyond the launch confirmation is required. The program will compile and run, printing the message, demonstrating a simple CUDA kernel launch.

*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloWorldKernel()
{
    // Empty kernel
}

int main(void)
{
    // Launch kernel with 1 block of 1 thread
    helloWorldKernel<<<1, 1>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Synchronize to ensure kernel has finished (optional here since kernel is empty)
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA device synchronize error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print confirmation message
    printf("Kernel launched successfully!\n");

    return 0;
}
