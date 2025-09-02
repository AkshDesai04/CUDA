/*
Aim: Profile the on-demand migration version vs. the prefetching version. Use Nsight Systems to visualize the page faults and data migrations.

Thinking: The goal of this CUDA program is to allow a user to compare the performance and runtime behavior of two common strategies when using unified memory on NVIDIA GPUs: 1) **On‑Demand Migration** – simply launch a kernel that accesses a large unified memory buffer without any explicit prefetching. 2) **Prefetching** – proactively move the entire buffer to the GPU device memory before the kernel launch with `cudaMemPrefetchAsync`. Unified memory (`cudaMallocManaged`) abstracts away explicit copy operations, but it relies on page‑fault–based migrations, which can introduce latency if data is not present on the device when accessed. By measuring and profiling both variants, one can observe the difference in page‑fault counts and migration traffic that Nsight Systems will record.

The program proceeds as follows:

1. Allocate a large buffer (`int *d_array`) using `cudaMallocManaged`.  
   The buffer size is chosen to be relatively large (e.g., 256 million integers ≈ 1 GB) so that it is likely to exceed the GPU’s memory capacity and trigger page migrations on-demand.  
2. Initialize the buffer on the host.  
3. Run the **on‑demand** version: launch a simple vector‑add kernel that increments each element.  
4. Run the **prefetch** version: first call `cudaMemPrefetchAsync` to bring the entire buffer to the GPU, wait for the prefetch to complete, then launch the same kernel.  
5. Measure execution times with `std::chrono` and print them.  
6. Wrap each variant with `cudaProfilerStart/Stop` so that Nsight Systems can capture a clear segment of activity for each case.  

The kernel is intentionally simple to focus on memory traffic rather than computational complexity. Users can compile this file with `nvcc` and run it under Nsight Systems (`nsys profile -o output ./a.out`). They will then be able to examine page‑fault events and data‑migration activity in the timeline view.

Below is the complete .cu file. No additional files or libraries are required.*/

#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

// Macro for checking CUDA errors
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Simple kernel that increments each element of an array
__global__ void incrementKernel(int *data, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1;
    }
}

// Runs the on-demand migration version
void runOnDemand(int *d_array, size_t N)
{
    printf("Running On‑Demand Migration Version...\n");

    // Timing start
    auto start = std::chrono::high_resolution_clock::now();

    cudaProfilerStart();

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    incrementKernel<<<blocks, threadsPerBlock>>>(d_array, N);
    CHECK_CUDA(cudaGetLastError());

    // Wait for kernel to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaProfilerStop();

    // Timing end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("On‑Demand time: %.6f seconds\n", elapsed.count());
}

// Runs the prefetching version
void runPrefetch(int *d_array, size_t N)
{
    printf("\nRunning Prefetching Version...\n");

    // Timing start
    auto start = std::chrono::high_resolution_clock::now();

    cudaProfilerStart();

    // Get current device
    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    // Prefetch entire array to GPU
    CHECK_CUDA(cudaMemPrefetchAsync(d_array, N * sizeof(int), device, 0));
    // Wait for prefetch to complete
    CHECK_CUDA(cudaDeviceSynchronize());

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    incrementKernel<<<blocks, threadsPerBlock>>>(d_array, N);
    CHECK_CUDA(cudaGetLastError());

    // Wait for kernel to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaProfilerStop();

    // Timing end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Prefetching time: %.6f seconds\n", elapsed.count());
}

int main(int argc, char *argv[])
{
    // Size of array: 256 million ints (~1 GB)
    const size_t N = 256 * 1000 * 1000;  // 256,000,000 elements

    printf("Allocating unified memory (%zu MB)...\n", N * sizeof(int) / (1024 * 1024));

    int *d_array = nullptr;
    CHECK_CUDA(cudaMallocManaged(&d_array, N * sizeof(int)));

    // Initialize array on host
    for (size_t i = 0; i < N; ++i) {
        d_array[i] = 0;
    }

    // Run on-demand version
    runOnDemand(d_array, N);

    // Run prefetching version
    runPrefetch(d_array, N);

    // Clean up
    CHECK_CUDA(cudaFree(d_array));

    printf("\nDone.\n");
    return 0;
}
