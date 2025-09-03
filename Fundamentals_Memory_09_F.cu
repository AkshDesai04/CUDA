```cpp
/*
Write a function that takes a boolean `use_pinned_memory` and performs the timed transfer, abstracting the allocation/deallocation logic.

Thought process:
- The goal is to encapsulate all the allocation, copy, and cleanup steps in a single function that can be reused for benchmarking or other purposes.
- The function should accept a `bool use_pinned_memory` flag to decide whether the host memory should be page‑locked (pinned) or regular pageable memory.
- We'll need to allocate host memory using either `cudaMallocHost` (for pinned) or `malloc` (for pageable). Correspondingly we must free with `cudaFreeHost` or `free`.
- Device memory will always be allocated with `cudaMalloc` and freed with `cudaFree`.
- Timing: Use CUDA events (`cudaEvent_t`) to accurately measure GPU copy time. Record a start event, perform `cudaMemcpy`, record a stop event, then compute elapsed time.
- Return the elapsed time in milliseconds. In case of any error, print an error message and return a negative value.
- The function will also print details of the transfer, such as bytes transferred and memory type used, for debugging purposes.
- After the transfer, all allocated memory is deallocated to avoid leaks.
- We'll provide a simple `main` that calls this function with both pinned and non‑pinned options to demonstrate usage.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Helper macro for checking CUDA errors
#define CUDA_CHECK(err)                                      \
    do {                                                     \
        cudaError_t _e = (err);                              \
        if (_e != cudaSuccess) {                             \
            std::cerr << "CUDA error at " << __FILE__        \
                      << ":" << __LINE__ << ": "            \
                      << cudaGetErrorString(_e) << std::endl; \
            std::exit(EXIT_FAILURE);                         \
        }                                                    \
    } while (0)

// Function that abstracts allocation, transfer, timing, and deallocation
double timedTransfer(bool use_pinned_memory, size_t size_in_bytes) {
    void* h_ptr = nullptr;
    void* d_ptr = nullptr;

    // Allocate host memory
    if (use_pinned_memory) {
        CUDA_CHECK(cudaMallocHost(&h_ptr, size_in_bytes));
    } else {
        h_ptr = std::malloc(size_in_bytes);
        if (!h_ptr) {
            std::cerr << "Failed to allocate pageable host memory." << std::endl;
            return -1.0;
        }
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_ptr, size_in_bytes));

    // Initialize host memory (optional)
    std::memset(h_ptr, 0xAA, size_in_bytes);

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Perform host-to-device copy
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size_in_bytes, cudaMemcpyHostToDevice));

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Compute elapsed time
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Clean up events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Deallocate memory
    if (use_pinned_memory) {
        CUDA_CHECK(cudaFreeHost(h_ptr));
    } else {
        std::free(h_ptr);
    }
    CUDA_CHECK(cudaFree(d_ptr));

    // Print transfer details
    std::cout << "Transferred " << size_in_bytes / (1024.0 * 1024.0)
              << " MB (" << (use_pinned_memory ? "Pinned" : "Pageable")
              << " host) in " << elapsed_ms << " ms." << std::endl;

    return static_cast<double>(elapsed_ms);
}

int main() {
    const size_t transfer_size = 256 * 1024 * 1024; // 256 MB

    std::cout << "Benchmarking with pageable host memory:" << std::endl;
    timedTransfer(false, transfer_size);

    std::cout << "\nBenchmarking with pinned host memory:" << std::endl;
    timedTransfer(true, transfer_size);

    return 0;
}
```