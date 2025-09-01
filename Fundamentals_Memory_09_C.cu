```cuda
/* 
Time both HtoD and DtoH transfers with pinned memory and compare to the pageable case.

Thinking:
- The goal is to benchmark host-to-device (HtoD) and device-to-host (DtoH) memory copy performance.
- We'll compare two scenarios: using pageable (regular malloc) memory on the host and using pinned (page-locked) memory (cudaMallocHost).
- Pinned memory allows the GPU to access host memory directly via DMA, so transfers should be faster.
- We'll allocate a sizable buffer (e.g., 10 million floats ≈ 40 MB) to make transfer times noticeable.
- For each scenario, we perform the copy multiple times (e.g., 100 iterations) and average the elapsed time using CUDA events.
- We'll use cudaMemcpy for the actual transfer. The events give millisecond precision timing.
- The program prints the average times for HtoD and DtoH for both pageable and pinned cases, allowing a clear comparison.
- Error checking is added via a macro to simplify CUDA API error handling.
- The code is self‑contained and can be compiled with nvcc.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

#define ITERATIONS 100

// Function to time a single cudaMemcpy operation
float timeMemcpy(void *src, void *dst, size_t size, cudaMemcpyKind kind, const char *desc) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(dst, src, size, kind));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

int main(void) {
    const size_t N = 10 * 1024 * 1024;           // 10 million floats
    const size_t bytes = N * sizeof(float);

    // Allocate device memory
    float *d_data = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_data, bytes));

    // Allocate pageable host memory
    float *h_pageable = (float *)malloc(bytes);
    if (!h_pageable) {
        fprintf(stderr, "Failed to allocate pageable host memory\n");
        exit(EXIT_FAILURE);
    }

    // Allocate pinned host memory
    float *h_pinned = NULL;
    CHECK_CUDA(cudaMallocHost((void **)&h_pinned, bytes));

    // Initialize host buffers (optional, not needed for timing but for correctness)
    for (size_t i = 0; i < N; ++i) {
        h_pageable[i] = (float)i;
        h_pinned[i] = (float)i;
    }

    // Variables to accumulate times
    float total_htoD_pageable = 0.0f;
    float total_htoD_pinned   = 0.0f;
    float total_dtoH_pageable = 0.0f;
    float total_dtoH_pinned   = 0.0f;

    // Warm up copies
    CHECK_CUDA(cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h_pageable, d_data, bytes, cudaMemcpyDeviceToHost));

    // Benchmark HtoD with pageable memory
    for (int i = 0; i < ITERATIONS; ++i) {
        float ms = timeMemcpy(h_pageable, d_data, bytes, cudaMemcpyHostToDevice, "HtoD pageable");
        total_htoD_pageable += ms;
    }

    // Benchmark HtoD with pinned memory
    for (int i = 0; i < ITERATIONS; ++i) {
        float ms = timeMemcpy(h_pinned, d_data, bytes, cudaMemcpyHostToDevice, "HtoD pinned");
        total_htoD_pinned += ms;
    }

    // Benchmark DtoH with pageable memory
    for (int i = 0; i < ITERATIONS; ++i) {
        float ms = timeMemcpy(d_data, h_pageable, bytes, cudaMemcpyDeviceToHost, "DtoH pageable");
        total_dtoH_pageable += ms;
    }

    // Benchmark DtoH with pinned memory
    for (int i = 0; i < ITERATIONS; ++i) {
        float ms = timeMemcpy(d_data, h_pinned, bytes, cudaMemcpyDeviceToHost, "DtoH pinned");
        total_dtoH_pinned += ms;
    }

    // Compute averages
    float avg_htoD_pageable = total_htoD_pageable / ITERATIONS;
    float avg_htoD_pinned   = total_htoD_pinned / ITERATIONS;
    float avg_dtoH_pageable = total_dtoH_pageable / ITERATIONS;
    float avg_dtoH_pinned   = total_dtoH_pinned / ITERATIONS;

    printf("Average transfer times over %d iterations (ms):\n", ITERATIONS);
    printf("HtoD pageable: %.3f ms\n", avg_htoD_pageable);
    printf("HtoD pinned:   %.3f ms\n", avg_htoD_pinned);
    printf("DtoH pageable: %.3f ms\n", avg_dtoH_pageable);
    printf("DtoH pinned:   %.3f ms\n", avg_dtoH_pinned);

    // Clean up
    free(h_pageable);
    CHECK_CUDA(cudaFreeHost(h_pinned));
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
```