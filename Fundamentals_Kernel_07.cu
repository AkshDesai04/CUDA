/*
Aim: Launch a kernel with a 2D block of threads, for example, 8x8 threads. Each thread should print its 2D thread index (`threadIdx.x`, `threadIdx.y`).

Thinking:
1. We need a CUDA C program that defines a kernel to print each thread's 2D indices.
2. We'll use `printf` inside the kernel. CUDA supports `printf` from device code if compiled with `-arch=sm_20` or higher and `--use_fast_math` is not relevant. We'll rely on default.
3. The kernel will simply call `printf("Thread (%d, %d)\n", threadIdx.x, threadIdx.y);`.
4. In `main`, we set a 2D block size of 8x8 threads (`dim3 block(8,8);`) and launch a single block (`dim3 grid(1,1);`).
5. After launching the kernel, we call `cudaDeviceSynchronize()` to ensure all device output is flushed before the program exits.
6. We'll include `<stdio.h>` for `printf`. No extra libraries or error checks are necessary for this simple demonstration, but we can optionally check for launch errors with `cudaGetLastError()`.
7. The program will be a single `.cu` file with the kernel and `main` function.
8. Since the prompt explicitly requires the aim line exactly, it will appear verbatim in the multiline comment. We'll then add our internal reasoning comments inside the same block comment, after the aim line.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printThreadIdx()
{
    // Each thread prints its 2D index
    printf("Thread (%d, %d)\n", threadIdx.x, threadIdx.y);
}

int main()
{
    // Define a 2D block of 8x8 threads
    dim3 blockDim(8, 8);
    // Launch a single block
    dim3 gridDim(1, 1);

    // Launch kernel
    printThreadIdx<<<gridDim, blockDim>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA device synchronization failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
