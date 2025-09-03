/*
Inside the kernel, calculate a flattened 1D thread index from the 2D indices: `int id_1d = threadIdx.y * blockDim.x + threadIdx.x;`. Print this value.

I will write a minimal CUDA program that launches a small 2D block of threads, each thread calculates its flattened 1D index using the provided formula, and prints that index using the device `printf`. The kernel will be launched with a single block of size 2×2 for simplicity. The host code will include basic error checking, launch the kernel, synchronize, and exit. The program will be fully self-contained in a single .cu file, ready to compile with `nvcc`. The goal is to demonstrate the calculation of a 1D thread index from 2D thread indices and to output that value from each thread.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Utility macro for checking CUDA errors */
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",            \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

/* Kernel that calculates a flattened 1D thread index from 2D indices and prints it */
__global__ void printFlattenedIndexKernel()
{
    /* Calculate the flattened 1D index from the 2D thread indices */
    int id_1d = threadIdx.y * blockDim.x + threadIdx.x;

    /* Print the thread's 2D indices and its computed 1D index */
    printf("Thread (%d, %d) -> Flattened 1D index: %d\n",
           threadIdx.x, threadIdx.y, id_1d);
}

int main(void)
{
    /* Define block and grid dimensions */
    dim3 blockDim(2, 2, 1);   // 2x2 block
    dim3 gridDim(1, 1, 1);    // single block

    /* Launch the kernel */
    printFlattenedIndexKernel<<<gridDim, blockDim>>>();
    CHECK_CUDA(cudaGetLastError());

    /* Wait for GPU to finish before accessing on host */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Reset the device (optional) */
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
