/*
Modify the program to launch the kernel with one block containing two threads.

Thinking:
- The requirement is to modify a CUDA program so that its kernel launch configuration consists of a single block with two threads.
- A minimal example will include a simple kernel that prints the thread index using `printf` from the device.
- The launch configuration will be specified as `<<<1, 2>>>`, which means 1 block and 2 threads per block.
- In `main` we will launch the kernel and then call `cudaDeviceSynchronize()` to wait for completion before exiting.
- The code will include necessary headers and will be self‑contained so that it can be compiled directly with `nvcc`.
- No additional supporting information will be printed or included; only the .cu source is provided.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void myKernel()
{
    int tid = threadIdx.x;
    printf("Hello from thread %d\n", tid);
}

int main(void)
{
    // Launch one block containing two threads
    myKernel<<<1, 2>>>();
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA sync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    return 0;
}