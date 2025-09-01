/*
Aim: Launch the printing kernel with one block of 10 threads. Observe the output.

Thinking:
- We need a .cu file that compiles with nvcc.
- The kernel will use printf to output from each thread.
- Launch configuration: gridDim=1, blockDim=10.
- After kernel launch, we need to synchronize.
- Use cudaDeviceSynchronize and check errors.
- In main, we can set device, etc.
- We'll include <cstdio> for host side.
- Kernel prints thread index.
- After synchronization, program exits.

This program will print from device. The order may not be deterministic but we will observe 10 lines. 
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void printKernel()
{
    int tid = threadIdx.x;
    printf("Hello from thread %d\n", tid);
}

int main()
{
    // Launch the kernel with one block of 10 threads
    printKernel<<<1, 10>>>();

    // Wait for the kernel to finish and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
