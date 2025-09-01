/*
Launch the printing kernel with 10 blocks of one thread. Observe the output.

Thinking:
I need to create a simple CUDA program that demonstrates how to launch a kernel
with a specific grid and block configuration – 10 blocks and a single thread
per block. The kernel will print a message that includes the block and thread
indices so that the output clearly shows which block executed the code.
CUDA `printf` is used for printing from device code; I must remember to call
`cudaDeviceSynchronize()` after the kernel launch to ensure all output is
flushed before the program exits. Error checking after kernel launch and after
synchronization will help catch any issues early. The code is written in C++ with
CUDA extensions, compiled with `nvcc`. It will compile to a .cu file. This
comment is intentionally placed at the top of the file, as requested, with the
exact aim statement from the prompt and a detailed explanation of the design
choices. */

#include <cstdio>
#include <cuda_runtime.h>

/* Simple kernel that prints its block and thread indices */
__global__ void printKernel()
{
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void)
{
    /* Launch kernel with 10 blocks, 1 thread per block */
    printKernel<<<10, 1>>>();
    CUDA_CHECK(cudaGetLastError());     // Check for launch errors

    /* Synchronize to wait for kernel completion and flush printf buffer */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Optionally, reset the device */
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
