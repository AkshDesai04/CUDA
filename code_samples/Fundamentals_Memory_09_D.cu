/*
Aim: What is the downside of using pinned memory? (It's a limited resource and reduces memory available to the OS and other applications).

Thinking:
1. Pinned (page-locked) memory is required for high‑performance asynchronous memory copies between host and device.
2. However, pinned memory consumes a special pool that is managed by the operating system and CUDA driver.
3. Allocating large amounts of pinned memory can exhaust this pool, causing allocation failures or impacting the OS’s ability to page memory.
4. It also reduces the amount of pageable memory available to other processes, which can lead to system thrashing or reduced performance for non‑CUDA applications.
5. To illustrate this, the following program:
   - Queries the maximum amount of pinned memory that can be allocated.
   - Attempts to allocate that amount plus a small extra to demonstrate failure.
   - Performs a simple kernel launch to show that pinned memory can be used for asynchronous copy.
   - Frees the memory and reports results.
6. The program uses CUDA runtime API and basic error checking. It prints the maximum pinned memory size, the success or failure of allocation attempts, and demonstrates a kernel that sums an array.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel to add two vectors
__global__ void vecAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(err)                                 \
    do {                                                \
        cudaError_t err__ = (err);                      \
        if (err__ != cudaSuccess) {                     \
            fprintf(stderr, "CUDA Error: %s (line %d)\n",\
                    cudaGetErrorString(err__), __LINE__); \
            exit(EXIT_FAILURE);                         \
        }                                               \
    } while (0)

int main(void)
{
    const int N = 1 << 20;          // 1M elements
    const size_t size = N * sizeof(float);

    // 1. Query maximum pinned memory
    size_t maxPinned;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxPinned, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0));
    printf("Maximum page-locked memory reported by device: %zu bytes\n", maxPinned);

    // Note: Some GPUs may not expose this attribute; fallback to a large number
    if (maxPinned == 0)
        maxPinned = 64 * 1024 * 1024; // 64MB as a conservative estimate

    // 2. Allocate pinned memory
    float *h_a, *h_b, *h_c;
    CUDA_CHECK(cudaMallocHost((void **)&h_a, size));
    CUDA_CHECK(cudaMallocHost((void **)&h_b, size));
    CUDA_CHECK(cudaMallocHost((void **)&h_c, size));

    // 3. Initialize host data
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 4. Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, size));
    CUDA_CHECK(cudaMalloc((void **)&d_b, size));
    CUDA_CHECK(cudaMalloc((void **)&d_c, size));

    // 5. Asynchronous copy from host to device using pinned memory
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream));

    // 6. Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vecAdd<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // 7. Copy result back to host
    CUDA_CHECK(cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream));

    // 8. Synchronize
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 9. Verify result
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c[i] - 3.0f) > 1e-5f) {
            ++errors;
            if (errors <= 10)
                printf("Mismatch at %d: %f\n", i, h_c[i]);
        }
    }
    if (errors == 0)
        printf("All results correct.\n");
    else
        printf("Total mismatches: %d\n", errors);

    // 10. Demonstrate pinned memory exhaustion
    printf("\nAttempting to allocate extra pinned memory beyond the reported maximum...\n");
    size_t attemptSize = maxPinned + 1; // one byte more than reported maximum
    float *extraPinned = NULL;
    cudaError_t err = cudaMallocHost((void **)&extraPinned, attemptSize);
    if (err == cudaSuccess) {
        printf("Unexpectedly succeeded in allocating %zu bytes of pinned memory.\n", attemptSize);
        cudaFreeHost(extraPinned);
    } else {
        printf("Failed to allocate %zu bytes of pinned memory: %s\n",
               attemptSize, cudaGetErrorString(err));
    }

    // 11. Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));
    CUDA_CHECK(cudaStreamDestroy(stream));

    printf("\nProgram completed.\n");
    return 0;
}
