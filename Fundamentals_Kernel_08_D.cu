/*
Aim of the program:
Launch a 3D grid of (2, 2, 2) blocks. Have each thread print its
3D block index (blockIdx.x, blockIdx.y, blockIdx.z).

My thinking:
1. We need a kernel that prints the block indices of the block it is running in.
2. Each thread will execute the same printf; the threadIdx can also be printed to
   distinguish which thread is doing the printing.
3. A minimal block dimension of (1,1,1) keeps the output concise while still
   demonstrating that each thread (only one per block in this case) prints its
   block index.
4. The grid dimension is set to (2,2,2) as requested.
5. The kernel is launched from the host main function, followed by
   cudaDeviceSynchronize() to ensure all prints are flushed.
6. Basic error checking is included for good practice.
7. CUDA runtime headers and stdio are included, and the code is written in a
   single .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Kernel that prints the block index for each thread */
__global__ void printBlockIdx()
{
    /* Each thread prints its own block index */
    printf("Thread (%d,%d,%d) in block (%d,%d,%d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z);
}

/* Helper macro for error checking */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

int main(void)
{
    /* Define a 3D grid of (2,2,2) blocks */
    dim3 gridDim(2, 2, 2);
    /* Define a 3D block of (1,1,1) threads (one thread per block) */
    dim3 blockDim(1, 1, 1);

    /* Launch the kernel */
    printBlockIdx<<<gridDim, blockDim>>>();

    /* Check for launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Wait for the GPU to finish */
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Kernel execution completed.\n");
    return 0;
}
