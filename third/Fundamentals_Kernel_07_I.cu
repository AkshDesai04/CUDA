/*
Modify the kernel so that threads with `threadIdx.x == threadIdx.y` (on the diagonal) print a special message.
Thought process:
- Use a 2D grid of threads to illustrate diagonal.
- In kernel, check if threadIdx.x == threadIdx.y and print special message via printf.
- Also print normal message for others.
- Use printf from device; compile with nvcc with -arch=sm_35 or later.
- In host, launch kernel with appropriate block and grid dims, say 4x4.
- Use cudaDeviceSynchronize to wait for completion.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",      \
                    __func__, __FILE__, __LINE__,                  \
                    cudaGetErrorString(err));                      \
            exit(EXIT_FAILURE);                                   \
        }                                                          \
    } while (0)

// Kernel that prints special message on the diagonal
__global__ void diagKernel()
{
    int x = threadIdx.x;
    int y = threadIdx.y;

    if (x == y)
    {
        printf("Thread (%d, %d) on diagonal!\n", x, y);
    }
    else
    {
        printf("Thread (%d, %d) normal.\n", x, y);
    }
}

int main()
{
    // Define a 4x4 block of threads
    dim3 blockDim(4, 4);
    dim3 gridDim(1, 1);

    // Launch the kernel
    diagKernel<<<gridDim, blockDim>>>();
    CUDA_CHECK(cudaGetLastError());

    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reset the device (optional)
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
