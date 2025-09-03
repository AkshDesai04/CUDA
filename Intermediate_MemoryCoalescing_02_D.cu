```cpp
/*
Aim of the program: This pattern ensures both global reads and global writes are fully coalesced.

Thinking:
1. The goal is to demonstrate a CUDA kernel that performs memory accesses in a way that guarantees full coalescing of global reads and writes. 
2. Full coalescing is achieved when consecutive threads access consecutive memory addresses in a contiguous array. 
3. A simple copy kernel where each thread copies one element from a source array to a destination array meets this requirement if the arrays are stored contiguously in memory.
4. The kernel will compute a global index for each thread (blockIdx.x * blockDim.x + threadIdx.x) and check bounds before performing the copy.
5. The host code will allocate two large arrays on the device, initialize the source array on the host, copy it to the device, launch the kernel with enough threads to cover the array, copy the result back, and verify correctness.
6. Error checking macros are used for clarity. Timing via cudaEvent can optionally be added to demonstrate performance, but it is omitted for brevity. 
7. The code is self‑contained, uses standard CUDA runtime APIs, and compiles to a .cu file that can be built with nvcc. 
8. All memory accesses are coalesced because each thread accesses a unique element at a stride of 1 in both source and destination arrays.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that copies data from src to dst using fully coalesced global accesses
__global__ void copyKernel(const float* __restrict__ src, float* __restrict__ dst, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // Both read and write are coalesced because idx is contiguous across threads
        dst[idx] = src[idx];
    }
}

int main()
{
    const size_t N = 1 << 24; // 16M elements (~64 MB per array)
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_src = (float*)malloc(bytes);
    float *h_dst = (float*)malloc(bytes);
    if (!h_src || !h_dst)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize source array
    for (size_t i = 0; i < N; ++i)
        h_src[i] = static_cast<float>(i);

    // Allocate device memory
    float *d_src = nullptr;
    float *d_dst = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_src, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_dst, bytes));

    // Copy source data to device
    CUDA_CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    copyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost));

    // Verify correctness
    bool success = true;
    for (size_t i = 0; i < N; ++i)
    {
        if (h_dst[i] != h_src[i])
        {
            fprintf(stderr, "Mismatch at index %zu: src=%f dst=%f\n",
                    i, h_src[i], h_dst[i]);
            success = false;
            break;
        }
    }
    if (success)
        printf("Copy successful: all elements match.\n");
    else
        printf("Copy failed: mismatched elements.\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    free(h_src);
    free(h_dst);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```