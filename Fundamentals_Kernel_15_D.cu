```cpp
/*
Aim of the program: 
Write a kernel that has a very long but finite loop. Call it without `cudaDeviceSynchronize`. The host program finishes almost instantly. Now add `cudaDeviceSynchronize`. The host program now waits for the kernel to complete.

Thinking:
To demonstrate the effect of `cudaDeviceSynchronize` we launch a very simple kernel that performs a long deterministic loop. The kernel is launched twice: once without waiting for completion, and once with an explicit synchronization call. By measuring the elapsed time between launch and the next host statement we can see the difference: the first case should return almost immediately (the kernel runs asynchronously in the background), whereas the second case should block until the kernel finishes. A single thread kernel that iterates a large number of times is sufficient to produce a noticeable delay on the GPU. We use `std::chrono` for timing and CUDA error checking macros for safety. The program is self‑contained and compiles to a single .cu file. 
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <iostream>

#define CUDA_CHECK(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

// Kernel that performs a very long but finite loop
__global__ void longLoopKernel(long long *output) {
    // Single thread performs the loop
    long long sum = 0;
    const long long iterations = 50000000LL; // 50 million iterations
    for (long long i = 0; i < iterations; ++i) {
        sum += i; // simple arithmetic to keep the loop active
    }
    // Store result to prevent compiler optimization
    output[0] = sum;
}

int main() {
    // Allocate a single element on device to store kernel result
    long long *d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(long long)));

    // ----- Launch without cudaDeviceSynchronize -----
    std::cout << "Launching kernel without synchronization..." << std::flush;
    auto start_no_sync = std::chrono::high_resolution_clock::now();

    // Launch kernel (single block, single thread)
    longLoopKernel<<<1, 1>>>(d_result);

    // No synchronization: host continues immediately
    auto end_no_sync = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_no_sync = end_no_sync - start_no_sync;
    std::cout << "\nDone (no sync). Elapsed time: " << elapsed_no_sync.count() << " seconds.\n";

    // ----- Launch with cudaDeviceSynchronize -----
    std::cout << "Launching kernel with synchronization..." << std::flush;
    auto start_sync = std::chrono::high_resolution_clock::now();

    longLoopKernel<<<1, 1>>>(d_result);

    // Explicitly wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end_sync = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_sync = end_sync - start_sync;
    std::cout << "\nDone (with sync). Elapsed time: " << elapsed_sync.count() << " seconds.\n";

    // Clean up
    CUDA_CHECK(cudaFree(d_result));
    return 0;
}
```