```cpp
/*
Aim of the program:
Compare the performance of `cudaMemset` vs. your custom zeroing kernel for a very large array. `cudaMemset` is usually highly optimized.

Thinking:
The goal is to demonstrate that the built‑in `cudaMemset` routine, which is heavily optimized by NVIDIA, can outperform a user‑defined kernel that simply writes zeros into device memory. To do this comparison we need:
1. Allocate a large device array – the larger the better, but we must stay within the GPU memory limits. 256 M integers (≈1 GB) is a good compromise for most GPUs.
2. Time a call to `cudaMemset(d_ptr, 0, size*sizeof(int))` using CUDA events for high‑resolution timing.
3. Time a custom kernel that sets each element to zero.  The kernel will launch a 1‑D grid with enough threads to cover the whole array.  Each thread will compute its global index and write zero if the index is in bounds.
4. Print the elapsed times for both methods and, optionally, compute a checksum to confirm correctness (though this step is not strictly necessary for the performance measurement).
5. Use simple error checking macros to keep the code clean.

The program is written in plain C (with a few C++ conveniences for `std::cout`) and should compile with `nvcc`.  No external libraries beyond the CUDA runtime are required.

Note: On GPUs with very large memory you can increase `ARRAY_SIZE` to test even larger arrays.  Make sure you have enough free memory to avoid out‑of‑memory errors.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// Error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error in " << #call << " (" << __FILE__       \
                      << ":" << __LINE__ << "): " << cudaGetErrorString(err)\
                      << std::endl;                                         \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Size of the array (number of integers).  256 million ints ≈ 1 GB
const size_t ARRAY_SIZE = (1 << 28); // 256 * 2^20

// Kernel to zero out an array
__global__ void zeroKernel(int *d_arr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = 0;
    }
}

int main() {
    std::cout << "Array size: " << ARRAY_SIZE << " ints (" 
              << static_cast<double>(ARRAY_SIZE) / (1<<20) << " Mi ints, "
              << static_cast<double>(ARRAY_SIZE * sizeof(int)) / (1<<30) 
              << " GiB)" << std::endl;

    // Allocate device memory
    int *d_arr;
    CUDA_CHECK(cudaMalloc(&d_arr, ARRAY_SIZE * sizeof(int)));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ----------------------------
    // 1) Time cudaMemset
    // ----------------------------
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemset(d_arr, 0, ARRAY_SIZE * sizeof(int)));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float timeMemset = 0;
    CUDA_CHECK(cudaEventElapsedTime(&timeMemset, start, stop));
    std::cout << "cudaMemset time: " << timeMemset << " ms" << std::endl;

    // ----------------------------
    // 2) Time custom zeroKernel
    // ----------------------------
    const int THREADS_PER_BLOCK = 256;
    int blocks = (ARRAY_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Warm up kernel launch to avoid first-launch overhead in timing
    zeroKernel<<<blocks, THREADS_PER_BLOCK>>>(d_arr, ARRAY_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, 0));
    zeroKernel<<<blocks, THREADS_PER_BLOCK>>>(d_arr, ARRAY_SIZE);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float timeKernel = 0;
    CUDA_CHECK(cudaEventElapsedTime(&timeKernel, start, stop));
    std::cout << "Custom zeroKernel time: " << timeKernel << " ms" << std::endl;

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_arr));

    std::cout << "Comparison:" << std::endl;
    std::cout << "cudaMemset is " << (timeKernel / timeMemset) << "× faster than the custom kernel." << std::endl;

    return 0;
}
```