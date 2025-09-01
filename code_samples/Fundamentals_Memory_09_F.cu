/*
Write a function that takes a boolean `use_pinned_memory` and performs the timed transfer, abstracting the allocation/deallocation logic.

Thinking:
- We need to benchmark host‑to‑device and device‑to‑host transfers for both pinned and pageable host memory.
- Pinned (page‑locked) memory is allocated with cudaMallocHost and can give higher transfer bandwidth.
- The function should allocate host memory (pinned or not), allocate device memory, copy data, optionally launch a simple kernel, copy back, time the transfer, print results, and free all resources.
- Use CUDA events to measure transfer times accurately.
- Include basic error checking with a macro.
- In main, we will call this function twice: once with pinned memory, once with pageable memory, to demonstrate the difference.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Simple kernel that just copies data (identity operation)
__global__ void copyKernel(int *d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] = d_data[idx];
    }
}

// Function that allocates/deallocates memory, performs transfer, times it, and cleans up.
void timedTransfer(bool use_pinned_memory, size_t elementCount) {
    const size_t bytes = elementCount * sizeof(int);
    int *h_data = nullptr;
    int *d_data = nullptr;

    // Allocate host memory
    if (use_pinned_memory) {
        CHECK_CUDA(cudaMallocHost(&h_data, bytes)); // pinned memory
    } else {
        h_data = (int *)malloc(bytes); // pageable memory
    }

    // Initialize host data
    for (size_t i = 0; i < elementCount; ++i) {
        h_data[i] = static_cast<int>(i);
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_data, bytes));

    // Create events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Host to Device copy
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // Optionally launch a simple kernel (to include device execution time)
    int threadsPerBlock = 256;
    int blocks = (elementCount + threadsPerBlock - 1) / threadsPerBlock;
    copyKernel<<<blocks, threadsPerBlock>>>(d_data, static_cast<int>(elementCount));
    CHECK_CUDA(cudaGetLastError());

    // Device to Host copy
    CHECK_CUDA(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Compute elapsed time
    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Print results
    printf("Transfer with %s memory: %zu elements (%zu bytes) in %f ms\n",
           use_pinned_memory ? "pinned" : "pageable",
           elementCount,
           bytes,
           milliseconds);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
    if (use_pinned_memory) {
        CHECK_CUDA(cudaFreeHost(h_data));
    } else {
        free(h_data);
    }
}

int main() {
    const size_t elementCount = 1 << 24; // ~16 million integers (~64 MB)

    // Timed transfer with pageable memory
    timedTransfer(false, elementCount);

    // Timed transfer with pinned memory
    timedTransfer(true, elementCount);

    return 0;
}
