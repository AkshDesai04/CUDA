/*
 * Aim: Pass an integer offset `d`. Have threads print `(threadIdx.x + d, threadIdx.y + d)`.
 *
 * Thinking:
 * 1. We need a CUDA kernel that receives an integer `d` and prints the
 *    offset indices for each thread. The kernel will use `printf` which
 *    is supported on devices of compute capability 2.0 or higher.
 * 2. The host program will parse `d` from the command line; if not
 *    supplied, it defaults to 0.
 * 3. For demonstration we launch a modest grid: 2 blocks × 2 blocks,
 *    each block containing 4 × 4 threads, giving 32 threads in total.
 * 4. We add basic error checking after the kernel launch and after
 *    device synchronization.
 * 5. The code is self‑contained and can be compiled with `nvcc`:
 *    `nvcc -o offset offset.cu`
 *
 * The program prints the offsets for all launched threads, each on a
 * separate line, in the format `(x, y)`.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Simple macro for CUDA error checking */
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

/* Kernel that prints (threadIdx.x + d, threadIdx.y + d) */
__global__ void printOffsetKernel(int d)
{
    printf("(%d, %d)\n", threadIdx.x + d, threadIdx.y + d);
}

int main(int argc, char *argv[])
{
    int d = 0;   /* default offset */

    /* Parse command line argument if provided */
    if (argc > 1) {
        d = atoi(argv[1]);
    }

    /* Define block and grid dimensions */
    dim3 threadsPerBlock(4, 4);  /* 4 x 4 = 16 threads per block */
    dim3 numBlocks(2, 2);        /* 2 x 2 = 4 blocks */

    /* Launch the kernel */
    printOffsetKernel<<<numBlocks, threadsPerBlock>>>(d);

    /* Check for launch errors */
    CHECK_CUDA(cudaGetLastError());

    /* Wait for kernel to finish */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Reset device before exit */
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
