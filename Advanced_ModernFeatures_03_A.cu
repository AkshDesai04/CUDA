/*
Write a parent kernel that launches a simple child kernel (e.g., a `printf`) with a configuration of 1 block and 1 thread.
My approach is to use CUDA dynamic parallelism, which allows a kernel to launch another kernel. The child kernel simply prints a message. The parent kernel launches the child with a 1×1 configuration. After launching the parent kernel from the host, I synchronize the device to ensure all kernel activity, including the child kernel, has completed before the program exits. I also include basic error checking for CUDA API calls. This example requires a GPU supporting compute capability 3.5 or higher.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",         \
                    __FUNCTION__, __FILE__, __LINE__,                 \
                    cudaGetErrorString(err));                         \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

/* Child kernel: prints a message */
__global__ void childKernel(void)
{
    /* Only one thread will execute this */
    printf("Hello from child kernel!\n");
}

/* Parent kernel: launches the child kernel */
__global__ void parentKernel(void)
{
    /* Launch child kernel with 1 block and 1 thread */
    childKernel<<<1, 1>>>();
    /* No need for device synchronization here; host will sync later */
}

int main(void)
{
    /* Launch the parent kernel with 1 block and 1 thread */
    parentKernel<<<1, 1>>>();
    /* Check for launch errors */
    CHECK_CUDA(cudaGetLastError());

    /* Synchronize to wait for all kernels (including child) to finish */
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("All kernels have finished execution.\n");

    return 0;
}
