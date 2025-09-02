/* 
Aim of the program: 
Compare the bandwidth of pageable vs. pinned (non-pageable) host memory. Perform the timing from the previous exercise. Then, repeat it but allocate the host memory using `cudaMallocHost` instead of `malloc`. Print both bandwidths and observe the difference. 

Thinking: 
- To measure bandwidth we need to transfer a large buffer from host to device and time it. 
- Use CUDA events for accurate GPU timing. 
- Allocate a sizable buffer (e.g. 100 MB) to get measurable times. 
- First test: use normal `malloc` to obtain pageable host memory, copy to the device, record the time, compute bandwidth. 
- Second test: use `cudaMallocHost` to allocate pinned host memory, copy to the device, record the time, compute bandwidth. 
- Print both bandwidths in GB/s. 
- Clean up all allocated memory. 
- Add simple error checking macro to keep code readable. 
- The program uses only C/C++ standard library functions and CUDA runtime API, so it compiles as a .cu file. 
*/ 

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

int main(void) {
    const size_t numBytes = 100 * 1024 * 1024; // 100 MB
    const size_t numElements = numBytes / sizeof(float);

    // Allocate device memory once
    float *d_ptr = NULL;
    CHECK_CUDA(cudaMalloc(&d_ptr, numBytes));

    // ---------- Test 1: Pageable host memory ----------
    float *h_pageable = (float*)malloc(numBytes);
    if (h_pageable == NULL) {
        fprintf(stderr, "Failed to allocate pageable host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host data (optional, not needed for bandwidth measurement)
    for (size_t i = 0; i < numElements; ++i) {
        h_pageable[i] = (float)i;
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d_ptr, h_pageable, numBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    double seconds = milliseconds / 1000.0;
    double bandwidth_pageable = (double)numBytes / (seconds * 1.0e9); // GB/s

    printf("Pageable host memory transfer:\n");
    printf("  Time: %.3f ms, Bandwidth: %.3f GB/s\n", milliseconds, bandwidth_pageable);

    // ---------- Test 2: Pinned host memory ----------
    float *h_pinned = NULL;
    CHECK_CUDA(cudaMallocHost(&h_pinned, numBytes));

    // Initialize pinned data (optional)
    for (size_t i = 0; i < numElements; ++i) {
        h_pinned[i] = (float)i;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d_ptr, h_pinned, numBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    seconds = milliseconds / 1000.0;
    double bandwidth_pinned = (double)numBytes / (seconds * 1.0e9); // GB/s

    printf("\nPinned host memory transfer:\n");
    printf("  Time: %.3f ms, Bandwidth: %.3f GB/s\n", milliseconds, bandwidth_pinned);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_ptr));
    free(h_pageable);
    CHECK_CUDA(cudaFreeHost(h_pinned));

    return 0;
}
