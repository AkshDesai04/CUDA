/*
Write a kernel where only thread 0 of a block prints a message (if (threadIdx.x == 0) { ... }). Launch with 128 threads.

Thinking:
- The requirement is to create a CUDA kernel that prints from thread 0 only.
- The kernel will check if(threadIdx.x == 0) and then call printf.
- We launch the kernel with 128 threads per block. Using a single block (blocks=1) keeps it simple.
- We need to include <stdio.h> for printf.
- After launching, we call cudaDeviceSynchronize() to wait for kernel completion before exiting main.
- No special error handling is added to keep the code minimal.
- The output is a simple .cu file containing the kernel, main, and necessary includes.
*/

#include <stdio.h>

__global__ void printKernel() {
    if (threadIdx.x == 0) {
        printf("Hello from thread 0 in block %d\n", blockIdx.x);
    }
}

int main() {
    const int threadsPerBlock = 128;
    const int blocks = 1;  // One block with 128 threads

    printKernel<<<blocks, threadsPerBlock>>>();

    cudaDeviceSynchronize();  // Ensure all printf output is flushed

    return 0;
}
