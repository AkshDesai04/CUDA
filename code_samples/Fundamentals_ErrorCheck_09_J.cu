```cu
/*
Aim: Nest try...catch blocks to show how different levels of an application could handle CUDA errors differently.

Thinking:
1. The goal is to illustrate error handling at three distinct layers:
   - The lowest level (kernel launch / device memory operations) throws an exception when a CUDA API call fails.
   - The middle level catches that exception, logs an informative message, and optionally attempts a fallback strategy (e.g., CPU fallback).
   - The top level (main) catches any remaining exceptions, reports them to the user, and exits cleanly.

2. To generate a realistic CUDA error we intentionally perform an invalid memory copy:
   * Allocate a device array of 100 floats.
   * Attempt to copy 200 floats from host to device using cudaMemcpy with a wrong size.
   This triggers `cudaErrorInvalidValue` which is detected by `cudaGetLastError` after the copy.

3. Error handling strategy:
   * `checkCudaErr` macro verifies the return status of CUDA calls and throws a `std::runtime_error` with an error message.
   * `launchKernel` (lowest level) performs all device operations, uses `checkCudaErr` after each CUDA call. If an exception propagates out, the calling function receives it.
   * `middleLayer` calls `launchKernel` inside a try-catch, logs the error, and decides whether to retry, fallback, or rethrow.
   * `main` wraps `middleLayer` in its own try-catch to catch any exceptions that survived the lower levels, prints a final error, and terminates.

4. The code is self‑contained in a single .cu file, uses only the CUDA runtime API and standard C++ exceptions, and demonstrates nested error handling across application layers.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <iostream>
#include <vector>

// Macro to check CUDA API return status and throw exception on failure
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::string msg = std::string("CUDA error in file '") + __FILE__ \
                             + "' in line " + std::to_string(__LINE__) +   \
                             ": " + cudaGetErrorString(err);                \
            throw std::runtime_error(msg);                                   \
        }                                                                    \
    } while (0)

// Simple kernel that just copies data (doesn't actually trigger an error)
__global__ void copyKernel(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Lowest level: performs device memory allocation, data transfer, kernel launch
void launchKernel(const std::vector<float>& hostSrc, std::vector<float>& hostDst) {
    int n = hostSrc.size();

    // Device pointers
    float *d_src = nullptr;
    float *d_dst = nullptr;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_src, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_dst, n * sizeof(float)));

    // Intentionally cause an error: copy more data than allocated
    size_t wrongSize = (n * 2) * sizeof(float); // double the intended size
    CHECK_CUDA(cudaMemcpy(d_src, hostSrc.data(), wrongSize, cudaMemcpyHostToDevice)); // This line throws

    // Launch kernel (will not be reached if the previous call throws)
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    copyKernel<<<blocks, threadsPerBlock>>>(d_src, d_dst, n);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA(cudaDeviceSynchronize()); // Ensure completion

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(hostDst.data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up
    cudaFree(d_src);
    cudaFree(d_dst);
}

// Middle level: calls launchKernel and handles exceptions
void middleLayer(const std::vector<float>& src, std::vector<float>& dst) {
    try {
        launchKernel(src, dst);
    } catch (const std::runtime_error& e) {
        std::cerr << "[Middle Layer] Caught CUDA exception: " << e.what() << std::endl;
        std::cerr << "[Middle Layer] Attempting fallback: Performing CPU copy instead." << std::endl;
        // Simple fallback: perform a CPU copy
        std::copy(src.begin(), src.end(), dst.begin());
        // Optionally rethrow if fallback fails or if you want to propagate error
        // throw;
    }
}

// Top level: calls middleLayer and handles remaining exceptions
int main() {
    const int N = 100;
    std::vector<float> hostSrc(N, 1.0f); // source data
    std::vector<float> hostDst(N, 0.0f); // destination data

    try {
        middleLayer(hostSrc, hostDst);
        std::cout << "[Main] Data transfer completed. First element: " << hostDst[0] << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Main] Unhandled exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```