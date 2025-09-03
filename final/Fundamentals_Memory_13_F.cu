```cpp
/*
Can cudaMallocHost (pinned memory allocation) also fail with a memory allocation error? Try it.

I want to demonstrate that cudaMallocHost can indeed fail when the system does not have enough contiguous host memory (including the pinned memory that CUDA reserves). To do this, I will attempt to allocate a large block of pinned memory, larger than what a typical system can provide. I will handle the return status from cudaMallocHost, check if it is cudaErrorMemoryAllocation, and print an informative message. After a successful allocation (if it ever succeeds) I will free the memory. The program is intentionally simple and uses the CUDA Runtime API. I also add a loop to try allocating progressively larger chunks until failure occurs, which provides a more robust test across systems with different memory capacities. This illustrates the failure scenario in a reproducible manner.
*/
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

int main(void) {
    size_t initialSize = 1ULL << 30; // 1 GiB
    size_t maxSize = 1ULL << 34;     // 16 GiB
    size_t step = 1ULL << 30;        // 1 GiB increments

    void* ptr = NULL;
    cudaError_t err;

    printf("Attempting to allocate pinned memory in increasing sizes...\n");

    for (size_t size = initialSize; size <= maxSize; size += step) {
        printf("Requesting %zu MiB of pinned memory... ", size / (1024 * 1024));
        fflush(stdout);

        err = cudaMallocHost(&ptr, size);
        if (err == cudaSuccess) {
            printf("Success.\n");
            // Immediately free it to reuse the memory for the next iteration
            err = cudaFreeHost(ptr);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to free pinned memory: %s\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
        } else if (err == cudaErrorMemoryAllocation) {
            printf("Failed: cudaErrorMemoryAllocation.\n");
            break;
        } else {
            printf("Failed: %s.\n", cudaGetErrorString(err));
            break;
        }
    }

    printf("Finished testing pinned memory allocation.\n");
    return 0;
}
```