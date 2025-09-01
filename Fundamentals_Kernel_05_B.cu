/*
Aim: Also print the thread's `blockIdx.x` and `threadIdx.x` next to its calculated global ID.

Thinking:
1. The task is to produce a CUDA C program that prints, for each thread in a kernel launch, the block index, thread index, and the computed global ID.
2. Use the device-side printf function, which is available on CUDA-enabled GPUs with compute capability >= 2.0.
3. The kernel will compute the global ID as:
       globalId = blockIdx.x * blockDim.x + threadIdx.x
   and then print the required information.
4. On the host side, we will configure a simple grid: e.g., 2 blocks with 5 threads each. This keeps the output manageable while demonstrating the indexing.
5. Add simple error checking after kernel launch and at the end of the program.
6. Include necessary headers and use the CUDA runtime API.
7. The program will be self‑contained in a single .cu file, ready to compile with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors */
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Kernel that prints blockIdx.x, threadIdx.x, and global ID */
__global__ void printThreadInfo()
{
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread blockIdx.x = %d, threadIdx.x = %d, global ID = %d\n",
           blockIdx.x, threadIdx.x, globalId);
}

int main(void)
{
    /* Define grid and block dimensions */
    dim3 blockDim(5);   // 5 threads per block
    dim3 gridDim(2);    // 2 blocks

    /* Launch the kernel */
    printThreadInfo<<<gridDim, blockDim>>>();
    CHECK_CUDA(cudaGetLastError()); // Check for launch errors

    /* Wait for the kernel to finish */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Reset the device */
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
