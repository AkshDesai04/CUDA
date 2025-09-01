/*
Can `cudaMallocHost` (pinned memory allocation) also fail with a memory allocation error? Try it.
The aim of this program is to test whether cudaMallocHost can fail due to a memory allocation error.
We repeatedly attempt to allocate large blocks of pinned host memory using cudaMallocHost until the call fails.
The program reports the error code and message when the allocation fails and cleans up any successfully allocated
memory. By requesting an unrealistically large allocation (e.g., 100 GB), we can trigger the failure on most systems,
demonstrating that cudaMallocHost can indeed return an allocation error, not just an out-of-memory condition
for device memory. This code also illustrates proper error handling and cleanup for pinned memory allocations.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Helper macro for checking CUDA errors
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

int main(void) {
    size_t allocationSize = (size_t)100 * 1024 * 1024 * 1024ULL; // 100 GB
    void *pinnedPtr = NULL;

    printf("Attempting to allocate %zu GB of pinned host memory using cudaMallocHost...\n",
           allocationSize / (1024 * 1024 * 1024ULL));

    cudaError_t err = cudaMallocHost(&pinnedPtr, allocationSize);

    if (err == cudaSuccess) {
        printf("Successfully allocated %zu GB of pinned memory.\n", allocationSize / (1024 * 1024 * 1024ULL));
        // Normally we would use the memory here, but we just free it immediately.
        cudaFreeHost(pinnedPtr);
        printf("Pinned memory freed successfully.\n");
    } else {
        printf("cudaMallocHost failed with error: %s (%d)\n",
               cudaGetErrorString(err), err);
    }

    // If we want to demonstrate repeated attempts until failure:
    // For large systems we skip this loop to avoid excessive memory usage.
    // Below is an example of allocating 1 GB blocks until failure.
    /*
    const size_t blockSize = 1ULL * 1024 * 1024 * 1024; // 1 GB
    size_t blocksAllocated = 0;
    void *blockPtr = NULL;

    while (true) {
        err = cudaMallocHost(&blockPtr, blockSize);
        if (err != cudaSuccess) {
            printf("Failed after allocating %zu blocks: %s (%d)\n",
                   blocksAllocated, cudaGetErrorString(err), err);
            break;
        }
        blocksAllocated++;
        printf("Allocated block %zu of 1 GB.\n", blocksAllocated);
    }

    // Free all allocated blocks
    // (In this example we only keep the last allocated pointer; 
    // real code would track all pointers in a list for freeing.)
    */

    return 0;
}
