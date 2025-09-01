```c
/* 
Launch with a 1D block of 32 threads. Have each thread print its warp ID (`threadIdx.x / 32`).

Thinking: 
The goal is to create a minimal CUDA program that launches a single block with exactly 32 threads. Each thread will compute its warp ID by performing an integer division of its thread index by 32. Since there are only 32 threads in the block, all threads will belong to warp 0 (i.e., threadIdx.x / 32 will evaluate to 0 for all threads). We still perform the division to demonstrate the intended calculation. 
We will use device-side printf, which requires the kernel to be compiled with the `-arch=sm_XX` flag that supports printf (most modern GPUs do). The main function will launch the kernel, synchronize, and perform basic error checking. 
The program will include the necessary headers and a small error-checking macro for readability. 
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void printWarpID()
{
    int warpID = threadIdx.x / 32;
    printf("Thread %d warp ID: %d\n", threadIdx.x, warpID);
}

int main()
{
    // Launch one block with 32 threads
    printWarpID<<<1, 32>>>();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return 0;
}
```