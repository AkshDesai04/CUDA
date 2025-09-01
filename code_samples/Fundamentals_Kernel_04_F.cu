/*
Launch with a 1D grid of 8 blocks, but give each block 4 threads.
Have only thread 0 of each block print the `blockIdx.x`.

Thought process:
- Need to use a CUDA kernel that uses blockIdx.x and threadIdx.x.
- Since we want only thread 0 of each block to print, we guard with if(threadIdx.x == 0).
- Use device printf, which requires compute capability >= 2.0; we assume the device supports it.
- Launch configuration: dim3 grid(8); dim3 block(4);
- After kernel launch, call cudaDeviceSynchronize to ensure all prints are flushed before program exits.
- Include error checking macros for robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple macro to check CUDA API return codes
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that prints blockIdx.x only from thread 0 of each block
__global__ void printBlockIdx()
{
    if (threadIdx.x == 0) {
        printf("BlockIdx.x: %d\n", blockIdx.x);
    }
}

int main(void)
{
    // Define grid and block dimensions
    dim3 gridDim(8);   // 8 blocks
    dim3 blockDim(4);  // 4 threads per block

    // Launch kernel
    printBlockIdx<<<gridDim, blockDim>>>();
    CHECK_CUDA(cudaGetLastError()); // Check for launch errors

    // Wait for kernel to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset device (optional)
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
