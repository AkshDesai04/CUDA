/*
 * Aim: Modify the previous 'Hello World' kernel to print a message from the GPU itself. The single thread in the kernel should use `printf` to print 'Hello from the GPU!'. Note that the output from the GPU might appear at a different time than host-side prints.
 *
 * Thinking: We want a simple CUDA program that launches a kernel with one block and one thread. Inside the kernel, we call `printf("Hello from the GPU!\n")`. After launching, the host calls `cudaDeviceSynchronize()` to ensure that all device output is flushed and the kernel has finished executing. We also check for errors after kernel launch and after synchronization. We need to include `<cstdio>` or `<stdio.h>` for the host printf. The kernel must compile with compute capability that supports device printf (>= 2.0). We keep the code minimal and portable. We'll also include a `__global__ void helloKernel()` definition. In main, we launch with `helloKernel<<<1,1>>>()` and then `cudaDeviceSynchronize()`. Finally we print a host message "Hello from the host!" to illustrate ordering differences. We also handle potential errors. This will produce output from the GPU and host.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Device kernel that prints a message
__global__ void helloKernel()
{
    // Only thread 0 prints
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("Hello from the GPU!\n");
    }
}

// Helper macro for CUDA error checking
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
    // Launch kernel with 1 block and 1 thread
    helloKernel<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());      // Check for launch errors

    // Wait for the GPU to finish and flush printf buffer
    CUDA_CHECK(cudaDeviceSynchronize());

    // Host message to illustrate ordering differences
    printf("Hello from the host!\n");

    return 0;
}
