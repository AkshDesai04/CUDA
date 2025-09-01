```c
/*
Aim of the program:
Calculate the total number of threads on the host (gridDim.x * blockDim.x) and pass it to the kernel.
Inside the kernel, print the global ID and the total number of threads.

Thinking:
The CUDA kernel needs to know how many total threads are launched so it can display that value.
We calculate totalThreads on the host by multiplying grid dimension and block dimension.
This value is passed to the kernel as a simple integer argument.
Inside the kernel, each thread calculates its global linear ID using
blockIdx.x * blockDim.x + threadIdx.x.
We then use the device printf function to output the thread's global ID and the total thread count.
Because device printf requires the device to have compute capability >= 2.0, we will compile
with `-rdc=true` and a suitable architecture flag (e.g. -arch=sm_35). The host code synchronizes
after kernel launch to ensure all prints are flushed before program exit.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

__global__ void printThreadInfo(int totalThreads)
{
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread %d: Global ID %d, Total Threads %d\n", threadIdx.x, globalID, totalThreads);
}

int main(void)
{
    // Define block and grid dimensions
    dim3 blockDim(256);
    dim3 gridDim(2);

    // Calculate total number of threads on the host
    int totalThreads = gridDim.x * blockDim.x;

    // Launch kernel with totalThreads passed as an argument
    printThreadInfo<<<gridDim, blockDim>>>(totalThreads);
    CHECK_CUDA(cudaGetLastError());

    // Synchronize to ensure all printf output is flushed
    CHECK_CUDA(cudaDeviceSynchronize());

    return 0;
}
```