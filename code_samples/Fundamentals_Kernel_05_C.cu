/*
Aim of the program: Calculate the total number of threads on the host (`gridDim.x * blockDim.x`) and pass it to the kernel. Inside the kernel, print the global ID and the total number of threads.

Thinking:
We want to create a simple CUDA program that demonstrates how to compute the total number of threads launched from the host side and then pass that information to the kernel. In CUDA, the grid and block dimensions are set on the host when invoking a kernel. While `gridDim.x` and `blockDim.x` are device-side variables that give the dimensions for each kernel invocation, the host can simply multiply the chosen block and grid sizes to obtain the same value. This total is then passed as a kernel argument.

On the device side, each thread calculates its unique global index using the standard formula:
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
We will also receive the total thread count from the host, so each thread can print its global ID together with that count. For printing from the device we use the built‑in `printf` function which is available on devices that support compute capability >= 2.0. We add error checking after the kernel launch and synchronize the device before exiting.

We choose modest block and grid sizes that are easy to reason about (e.g. 256 threads per block and 4 blocks, giving 1024 threads). The program will compile with nvcc and run, printing 1024 lines, one per thread.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel that prints global ID and total number of threads
__global__ void printThreadInfo(int totalThreads)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread %d: Total threads = %d\n", gid, totalThreads);
}

int main(void)
{
    // Define grid and block dimensions
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = 4;

    // Compute total number of threads on the host
    int totalThreads = blocksPerGrid * threadsPerBlock;

    printf("Launching kernel with %d blocks of %d threads each (%d total threads).\n",
           blocksPerGrid, threadsPerBlock, totalThreads);

    // Launch kernel
    printThreadInfo<<<blocksPerGrid, threadsPerBlock>>>(totalThreads);

    // Check for launch errors
    CHECK_CUDA(cudaGetLastError());

    // Wait for GPU to finish before accessing on host
    CHECK_CUDA(cudaDeviceSynchronize());

    return 0;
}
