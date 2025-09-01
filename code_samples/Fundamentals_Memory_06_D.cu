```cpp
/*
 * Aim of the program:
 * Demonstrate a different kind of leak: allocate with `cudaMallocHost` (pinned memory) and forget to free with `cudaFreeHost`.
 *
 * Thinking process:
 * 1. The goal is to illustrate how failing to release pinned memory leads to a leak that persists across kernel launches and program execution.
 * 2. We will allocate a chunk of pinned memory using `cudaMallocHost` inside a loop, and deliberately omit the `cudaFreeHost` call.
 * 3. To make the effect visible, we will query the amount of free device memory before and after each allocation using `cudaMemGetInfo`. This will show the decrease in free memory after each pinned allocation.
 * 4. We will also perform a trivial kernel launch (e.g., adding two vectors) to demonstrate that the program can still run kernels while the pinned memory leak persists.
 * 5. After the loop, we will print a final statement indicating that the pinned memory has not been freed.
 * 6. The program does not free the allocated pinned memory, thereby causing a memory leak on the host side (pinned memory remains reserved).
 * 7. The code is self-contained, uses standard CUDA runtime API, and can be compiled with `nvcc`. It includes error checking for CUDA API calls.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring> // for memset

// Simple vector addition kernel
__global__ void vecAdd(const float* a, const float* b, float* c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " : " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main()
{
    const int iterations = 5;
    const size_t vecSize = 1 << 20;          // 1 million elements
    const size_t vecBytes = vecSize * sizeof(float);

    // Allocate device memory once
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, vecBytes));
    CUDA_CHECK(cudaMalloc(&d_b, vecBytes));
    CUDA_CHECK(cudaMalloc(&d_c, vecBytes));

    // Initialize device vectors with some values
    std::vector<float> h_a(vecSize, 1.0f);
    std::vector<float> h_b(vecSize, 2.0f);
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), vecBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), vecBytes, cudaMemcpyHostToDevice));

    // Query free memory before starting
    size_t freePrev, total;
    CUDA_CHECK(cudaMemGetInfo(&freePrev, &total));
    std::cout << "Initial free device memory: " << freePrev / (1024 * 1024) << " MB" << std::endl;

    for (int i = 0; i < iterations; ++i)
    {
        // Allocate pinned host memory but do NOT free it
        void* pinnedPtr;
        size_t pinnedSize = 10 * 1024 * 1024; // 10 MB
        CUDA_CHECK(cudaMallocHost(&pinnedPtr, pinnedSize));

        // Fill pinned memory with dummy data (optional)
        std::memset(pinnedPtr, 0, pinnedSize);

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (vecSize + threadsPerBlock - 1) / threadsPerBlock;
        vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, vecSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check device memory usage after allocation
        size_t freeAfter, totalAfter;
        CUDA_CHECK(cudaMemGetInfo(&freeAfter, &totalAfter));
        std::cout << "Iteration " << i + 1
                  << " - free device memory: " << freeAfter / (1024 * 1024) << " MB"
                  << " (decrease: " << (freePrev - freeAfter) / (1024 * 1024)
                  << " MB)" << std::endl;

        freePrev = freeAfter;
        // Intentionally do NOT free pinned memory to create a leak
    }

    std::cout << "Pinned memory allocations completed. No cudaFreeHost calls were made, resulting in a leak." << std::endl;

    // Clean up device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
```