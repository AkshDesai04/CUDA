/*
Time both HtoD and DtoH transfers with pinned memory and compare to the pageable case.

Thinking:
The goal is to demonstrate the difference in transfer speed between pageable (regular malloc) and pinned (page‑locked) host memory for both host‑to‑device (HtoD) and device‑to‑host (DtoH) copies.  To achieve this we:
1. Allocate a host array in two forms:
   - Pageable: standard malloc/free.
   - Pinned: cudaMallocHost/cudaFreeHost.
2. Allocate a corresponding device array with cudaMalloc.
3. Initialize the pageable buffer with some data, then copy it into the pinned buffer so that both contain identical content.
4. For each transfer direction (HtoD, DtoH) and each buffer type (pageable, pinned) perform a number of repeated copies (e.g. 100 iterations) while timing with CUDA events.  The events are recorded before and after each cudaMemcpy call, synchronized, and the elapsed time is accumulated to compute an average.
5. After the loops, print the average times for each case.
6. Clean up all allocated memory.
The code checks for CUDA errors after each CUDA API call to catch failures early.  The buffer size chosen is 1M floats (≈4 MB) which is large enough to give measurable transfer times but small enough to avoid exhausting system memory.  The program compiles with nvcc and is self‑contained in a single .cu file. 
*/

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define CHECK_CUDA(call)                                        \
    do {                                                        \
        cudaError_t err = call;                                 \
        if (err != cudaSuccess) {                               \
            std::cerr << "CUDA error in " << __FILE__ << ":"    \
                      << __LINE__ << " : " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

int main()
{
    const size_t num_elements = 1 << 20; // 1,048,576 floats (~4 MB)
    const size_t bytes = num_elements * sizeof(float);
    const int iterations = 100; // number of copies per test

    // Host pageable memory
    float* h_pageable = (float*)malloc(bytes);
    if (!h_pageable) {
        std::cerr << "Failed to allocate pageable host memory." << std::endl;
        return EXIT_FAILURE;
    }

    // Host pinned memory
    float* h_pinned = nullptr;
    CHECK_CUDA(cudaMallocHost((void**)&h_pinned, bytes));

    // Initialize pageable buffer with sample data
    for (size_t i = 0; i < num_elements; ++i) {
        h_pageable[i] = static_cast<float>(i);
    }
    // Copy data to pinned buffer so both have same content
    std::memcpy(h_pinned, h_pageable, bytes);

    // Device memory
    float* d_device = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_device, bytes));

    // CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float elapsed_pageable_HtoD = 0.0f;
    float elapsed_pinned_HtoD   = 0.0f;
    float elapsed_pageable_DtoH = 0.0f;
    float elapsed_pinned_DtoH   = 0.0f;

    // ------- HtoD ------- //
    // Pageable
    for (int i = 0; i < iterations; ++i) {
        CHECK_CUDA(cudaEventRecord(start, 0));
        CHECK_CUDA(cudaMemcpy(d_device, h_pageable, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        elapsed_pageable_HtoD += ms;
    }

    // Pinned
    for (int i = 0; i < iterations; ++i) {
        CHECK_CUDA(cudaEventRecord(start, 0));
        CHECK_CUDA(cudaMemcpy(d_device, h_pinned, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        elapsed_pinned_HtoD += ms;
    }

    // ------- DtoH ------- //
    // Pageable
    for (int i = 0; i < iterations; ++i) {
        CHECK_CUDA(cudaEventRecord(start, 0));
        CHECK_CUDA(cudaMemcpy(h_pageable, d_device, bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        elapsed_pageable_DtoH += ms;
    }

    // Pinned
    for (int i = 0; i < iterations; ++i) {
        CHECK_CUDA(cudaEventRecord(start, 0));
        CHECK_CUDA(cudaMemcpy(h_pinned, d_device, bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        elapsed_pinned_DtoH += ms;
    }

    // Compute averages
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nAverage transfer times over " << iterations << " iterations (ms):\n";
    std::cout << "Pageable HtoD: " << elapsed_pageable_HtoD / iterations << " ms\n";
    std::cout << "Pinned   HtoD: " << elapsed_pinned_HtoD   / iterations << " ms\n";
    std::cout << "Pageable DtoH: " << elapsed_pageable_DtoH / iterations << " ms\n";
    std::cout << "Pinned   DtoH: " << elapsed_pinned_DtoH   / iterations << " ms\n";

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_device));
    CHECK_CUDA(cudaFreeHost(h_pinned));
    free(h_pageable);

    return 0;
}
