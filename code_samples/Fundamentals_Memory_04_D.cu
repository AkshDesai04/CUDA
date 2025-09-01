```cuda
/*
Compare the performance of `cudaMemset` vs. your custom zeroing kernel for a very large array. `cudaMemset` is usually highly optimized.

[Thinking]
To benchmark the two approaches we allocate a very large array on the device (1 GB for 32‑bit integers).  CUDA’s `cudaMemset` writes a constant byte pattern to a contiguous memory block, and is highly optimized by the driver, often using DMA and hardware prefetch.  The custom kernel will simply launch many threads, each writing a zero value to a unique element.  We will use CUDA events for accurate timing and compare the elapsed times.  Both approaches will be verified by reading back a few elements after each operation to ensure correctness.  We choose `int` as the data type because it maps directly to 4‑byte elements and avoids padding issues.  The kernel uses a simple 1D grid with 256 threads per block, which is a common choice for memory‑bound workloads.  We also perform error checking after each CUDA call to catch any failures.  The program prints the size of the array, the timing of `cudaMemset`, the timing of the custom kernel, and the difference in time.  This provides a clear, reproducible comparison of the two methods.
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

// Custom kernel to zero an array of ints
__global__ void zeroKernel(int* d_arr, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        d_arr[idx] = 0;
    }
}

// Helper macro for error checking
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            std::cerr << "CUDA error at " << __FILE__ << ":"       \
                      << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

int main()
{
    const size_t numElements = 1ULL << 28; // 256M ints => ~1GB
    const size_t sizeBytes   = numElements * sizeof(int);

    std::cout << "Allocating array of " << numElements
              << " elements (" << sizeBytes / (1024.0 * 1024.0 * 1024.0)
              << " GB)." << std::endl;

    int* d_arr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, sizeBytes));

    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 1. Measure cudaMemset
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemset(d_arr, 0, sizeBytes));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msMemset = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msMemset, start, stop));
    std::cout << "cudaMemset time: " << msMemset << " ms" << std::endl;

    // Verify a few elements are zero
    int h_val = 1;
    CUDA_CHECK(cudaMemcpy(&h_val, d_arr, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_val != 0) {
        std::cerr << "Verification failed after cudaMemset." << std::endl;
        return EXIT_FAILURE;
    }

    // 2. Measure custom zeroing kernel
    // Reset array to non-zero (e.g., all ones) to avoid caching effects
    CUDA_CHECK(cudaMemset(d_arr, 0xFF, sizeBytes)); // set to non-zero pattern

    int threadsPerBlock = 256;
    int blocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaEventRecord(start, 0));
    zeroKernel<<<blocks, threadsPerBlock>>>(d_arr, numElements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msKernel = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msKernel, start, stop));
    std::cout << "Custom kernel time: " << msKernel << " ms" << std::endl;

    // Verify a few elements are zero
    CUDA_CHECK(cudaMemcpy(&h_val, d_arr, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_val != 0) {
        std::cerr << "Verification failed after custom kernel." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Time difference: " << msKernel - msMemset << " ms" << std::endl;

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_arr));

    return EXIT_SUCCESS;
}
```