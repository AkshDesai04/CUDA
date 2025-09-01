```cpp
/* 
Aim of the program:
Modify the formula for a 2D grid of 1D blocks. Launch with a grid of (4, 2) blocks, each with 16 threads. The global ID formula is more complex. Start by calculating a unique block ID: `int blockId = blockIdx.y * gridDim.x + blockIdx.x;`. Then calculate the global thread ID.

Thinking:
- A 2D grid means blockIdx.x ranges 0..gridDim.x-1 and blockIdx.y ranges 0..gridDim.y-1.
- To get a unique linear block ID we multiply the y index by the number of blocks in x and add the x index: blockId = blockIdx.y * gridDim.x + blockIdx.x.
- Each block has a 1D arrangement of threads: threadIdx.x from 0..blockDim.x-1.
- The global thread ID is computed by scaling the block ID by the number of threads per block and adding the local thread ID: globalId = blockId * blockDim.x + threadIdx.x.
- We'll launch with grid dimensions (4,2) and block dimension 16, so the total number of threads is 4*2*16 = 128.
- The kernel will write each thread's global ID into an array that is later copied back to the host for printing.
- Error checking macros are included for CUDA API calls to simplify debugging.
- The program is self-contained and can be compiled with `nvcc -o grid_id grid_id.cu` and executed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel that calculates global thread IDs using a 2D grid of 1D blocks
__global__ void computeGlobalId(int *globalIds)
{
    // Compute a unique block ID in a flattened grid
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;

    // Local thread index within the block
    int threadId = threadIdx.x;

    // Global thread ID across the entire grid
    int globalId = blockId * blockDim.x + threadId;

    // Compute linear index into the output array
    int linearIdx = blockId * blockDim.x + threadId;

    // Store the global ID
    globalIds[linearIdx] = globalId;
}

int main(void)
{
    const int threadsPerBlock = 16;
    const dim3 gridDim(4, 2);               // 4 blocks in x, 2 blocks in y
    const int numBlocks = gridDim.x * gridDim.y;
    const int totalThreads = numBlocks * threadsPerBlock;

    // Allocate host memory
    int *h_globalIds = (int*)malloc(totalThreads * sizeof(int));
    if (h_globalIds == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_globalIds;
    CUDA_CHECK(cudaMalloc((void**)&d_globalIds, totalThreads * sizeof(int)));

    // Launch kernel
    computeGlobalId<<<gridDim, threadsPerBlock>>>(d_globalIds);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_globalIds, d_globalIds,
                          totalThreads * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Print the mapping from linear index to global ID
    printf("Thread mapping (linearIdx -> globalId):\n");
    for (int i = 0; i < totalThreads; ++i) {
        printf("%5d -> %5d\n", i, h_globalIds[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_globalIds));
    free(h_globalIds);

    return EXIT_SUCCESS;
}
```