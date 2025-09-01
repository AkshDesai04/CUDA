/*
Modify the `printf` to include the thread's ID: `printf("Hello from thread %d!\n", threadIdx.x);`. Launch with 8 threads.

Thinking:
- The goal is to adjust the kernel's printf to show the thread's ID using threadIdx.x.
- The kernel should be launched with a single block containing 8 threads as specified.
- After launching, synchronize the device to ensure all prints complete before the program exits.
- Include necessary headers and check for errors if desired.
- The program is minimal and demonstrates CUDA's device printf functionality.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel()
{
    printf("Hello from thread %d!\n", threadIdx.x);
}

int main()
{
    // Launch kernel with 1 block of 8 threads
    helloKernel<<<1, 8>>>();
    // Wait for device to finish
    cudaDeviceSynchronize();

    return 0;
}