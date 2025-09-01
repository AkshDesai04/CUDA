/*
Aim: Combine the concepts of the previous two questions. Launch a kernel with a grid of 4 blocks, each containing 16 threads. Each thread must calculate and print its unique 'global ID' using the formula: int id = blockIdx.x * blockDim.x + threadIdx.x;

Thinking:
1. The program will use CUDA C (C with CUDA extensions). 
2. We'll include <stdio.h> for the host printf and CUDA runtime API. 
3. Define a simple kernel `printIDs` that each thread executes. Inside the kernel, compute the global thread ID using the provided formula and print it with `printf`. 
4. On the host, configure the kernel launch with 4 blocks and 16 threads per block. 
5. After launching, we must synchronize the device (`cudaDeviceSynchronize()`) to ensure all prints are flushed before the program exits. 
6. We'll add minimal error checking for the kernel launch and synchronization, but keep the code concise. 
7. The code is intended to be compiled with `nvcc` and run on a CUDA-capable GPU. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printIDs()
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread %d\n", id);
}

int main()
{
    // Launch kernel with 4 blocks and 16 threads per block
    printIDs<<<4, 16>>>();
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish before accessing results
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
