/*
Calculate the global ID starting from 1 instead of 0.

Thinking:
The goal is to write a CUDA C program that demonstrates how to compute a
"global ID" for each thread, but with the numbering starting at 1 rather
than the usual 0-based index. In CUDA, a thread’s global index is usually
computed as:

    int gid = blockIdx.x * blockDim.x + threadIdx.x;

This yields a 0-based index. To shift it to start at 1, we simply add 1:

    int gid1 = blockIdx.x * blockDim.x + threadIdx.x + 1;

We will write a simple kernel that prints each thread’s global ID
starting from 1. For demonstration purposes we will launch a single
block with a modest number of threads (e.g., 64).  The kernel will use
`printf` to output the thread’s local and global IDs.  After kernel
launch, the host will call `cudaDeviceSynchronize()` to ensure all
output is flushed before the program exits.

This program is self‑contained and can be compiled with:

    nvcc -o global_id global_id.cu

and run with:

    ./global_id

It will print something like:

    Thread 0: global ID 1
    Thread 1: global ID 2
    ...
    Thread 63: global ID 64

The code below follows standard CUDA conventions and includes minimal
error checking for clarity.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void printGlobalID(int numThreads)
{
    int localIdx = threadIdx.x;
    // Compute 1-based global ID
    int globalID = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (globalID <= numThreads) {
        printf("Thread %d: global ID %d\n", localIdx, globalID);
    }
}

int main()
{
    const int threadsPerBlock = 64;
    const int numBlocks = 1;
    const int totalThreads = threadsPerBlock * numBlocks;

    // Launch kernel
    printGlobalID<<<numBlocks, threadsPerBlock>>>(totalThreads);

    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
