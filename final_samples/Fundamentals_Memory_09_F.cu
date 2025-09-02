```cpp
/*
Aim: Write a function that takes a boolean `use_pinned_memory` and performs the timed transfer, abstracting the allocation/deallocation logic.

Thinking:
1. The function must accept a bool flag `use_pinned_memory`.  
2. It should allocate a host buffer either with cudaMallocHost (pinned) or regular malloc (pageable).  
3. Allocate a device buffer of the same size using cudaMalloc.  
4. Initialize the host buffer with some data (e.g., sequential integers) to have a meaningful copy.  
5. Use CUDA events (cudaEventCreate, Record, Synchronize) to time the copy from host to device (`cudaMemcpyHtoD`).  
6. Optionally, copy back from device to host to fully exercise transfer, but the requirement only mentions timed transfer.  
7. After timing, clean up: free device memory, free host memory (cudaFreeHost if pinned, free if pageable), destroy events.  
8. Return the elapsed time in milliseconds or print it.  
9. Wrap the logic in a helper function `timedTransfer(bool use_pinned_memory)` that returns the elapsed time.  
10. Provide a `main` that calls this function with both options to demonstrate usage.  
11. Add necessary includes, error checking macros, and compile‑friendly code.  
12. The file will be a valid .cu source file that can be compiled with nvcc.  
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",         \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Timed transfer function
float timedTransfer(bool use_pinned_memory) {
    const size_t N = 1 << 20;          // 1M elements
    const size_t bytes = N * sizeof(int);
    int *h_data = nullptr;
    int *d_data = nullptr;

    // Allocate host memory
    if (use_pinned_memory) {
        CUDA_CHECK(cudaMallocHost((void**)&h_data, bytes));
    } else {
        h_data = (int*)malloc(bytes);
        if (!h_data) {
            fprintf(stderr, "Failed to allocate pageable host memory\n");
            exit(EXIT_FAILURE);
        }
    }

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = static_cast<int>(i);
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_data, bytes));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Perform transfer host to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    if (use_pinned_memory) {
        CUDA_CHECK(cudaFreeHost(h_data));
    } else {
        free(h_data);
    }

    return ms;
}

int main() {
    float time_pinned = timedTransfer(true);
    printf("Pinned memory transfer time: %f ms\n", time_pinned);

    float time_pageable = timedTransfer(false);
    printf("Pageable memory transfer time: %f ms\n", time_pageable);

    return 0;
}
```