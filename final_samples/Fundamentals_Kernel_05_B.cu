```c
/*
Aim of the program:
Also print the thread's `blockIdx.x` and `threadIdx.x` next to its calculated global ID.

Thinking:
- The program will use CUDA C to launch a kernel that runs on the GPU.
- Inside the kernel, each thread calculates its global index as
  globalID = blockIdx.x * blockDim.x + threadIdx.x.
- The kernel will then print, using device printf, the block index, thread index,
  and the computed global ID.
- In the host code, we set up a simple grid and block configuration,
  launch the kernel, synchronize, and check for errors.
- Since device printf requires compute capability 2.0 or higher, we assume
  the target GPU supports it.
- We will compile this file as a .cu file with nvcc, for example:
  nvcc -arch=sm_35 -o print_ids print_ids.cu
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printThreadInfo(void)
{
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;
    printf("BlockIdx.x: %d, ThreadIdx.x: %d, GlobalID: %d\n",
           blockIdx.x, threadIdx.x, globalID);
}

int main(void)
{
    // Define grid and block dimensions
    const int threadsPerBlock = 16;
    const int blocksPerGrid = 4;

    // Launch kernel
    printThreadInfo<<<blocksPerGrid, threadsPerBlock>>>();

    // Synchronize to ensure all printf output is flushed
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset device (optional)
    cudaDeviceReset();

    return 0;
}
```