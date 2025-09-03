```cpp
/*
Modify the formula for a 2D grid of 1D blocks. Launch with a grid of (4, 2) blocks, each with 16 threads. The global ID formula is more complex. Start by calculating a unique block ID: `int blockId = blockIdx.y * gridDim.x + blockIdx.x;`. Then calculate the global thread ID.

Thinking:
1. We need to support a 2D grid layout where each block is 1D (i.e., has only one dimension of threads).
2. The block ID must be unique across the 2D grid. The formula `blockIdx.y * gridDim.x + blockIdx.x` correctly maps the (x,y) coordinates of a block to a single linear index.
3. Once we have the unique block ID, we can compute the global thread ID by scaling the block ID by the number of threads per block and adding the thread's local index (`threadIdx.x`).
4. The kernel will simply write the computed global ID into an array so we can verify that the IDs are unique and continuous.
5. The host code will set up a grid of (4, 2) blocks and each block will contain 16 threads. That gives a total of 4*2*16 = 128 threads.
6. We'll allocate memory on the device, launch the kernel, copy back the results, and print a few of them to confirm correct execution.
7. Standard error checking macros will be used for CUDA API calls.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

__global__ void computeGlobalIds(int *output, int totalThreads) {
    // Compute unique block ID for a 2D grid of 1D blocks
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;

    // Compute global thread ID
    int globalId = blockId * blockDim.x + threadIdx.x;

    if (globalId < totalThreads) {
        output[globalId] = globalId;
    }
}

int main(void) {
    // Define grid and block dimensions
    dim3 blockDim(16);          // 16 threads per block (1D)
    dim3 gridDim(4, 2);         // 4 blocks in x, 2 blocks in y

    const int totalThreads = gridDim.x * gridDim.y * blockDim.x;
    const size_t size = totalThreads * sizeof(int);

    // Allocate host memory
    int *h_output = (int *)malloc(size);
    if (!h_output) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_output;
    CUDA_CHECK(cudaMalloc((void **)&d_output, size));

    // Launch kernel
    computeGlobalIds<<<gridDim, blockDim>>>(d_output, totalThreads);
    CUDA_CHECK(cudaGetLastError());      // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Print first 10 global IDs as a sanity check
    printf("First 10 global thread IDs:\n");
    for (int i = 0; i < 10 && i < totalThreads; ++i) {
        printf("h_output[%d] = %d\n", i, h_output[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_output));
    free(h_output);

    return 0;
}
```