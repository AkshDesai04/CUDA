```cuda
/*
Aim of the program: Profile the on-demand migration version vs. the prefetching version. Use Nsight Systems to visualize the page faults and data migrations.

Thinking:
- The goal is to provide a minimal, self‑contained CUDA program that demonstrates two execution strategies
  for Unified Memory (UM) data: 
    1) On‑Demand migration – the data is allocated with cudaMallocManaged and accessed by the kernel
       without any explicit prefetching. The CUDA runtime will migrate pages to the device lazily as
       they are touched.
    2) Prefetching – before launching the kernel we call cudaMemPrefetchAsync to bring the data
       to the GPU. This should reduce the number of page faults during kernel execution.
- To make the difference visible we will allocate a relatively large array (e.g. 64 MB) and
  perform a kernel that touches every element. Because the array spans many GPU memory pages,
  the on‑demand version will generate many page faults as the device fetches each page.
- We will time the kernel execution with CUDA events and print the elapsed time for each strategy.
  Nsight Systems can be used to capture the program and visualize the page faults and data migrations
  by running: `nsys profile -t cuda,cuda_api --output myrun ./unified_memory_profile`.
- The program is deliberately simple, with no complex logic, so that the user can easily modify
  the data size or kernel to observe the behavior in Nsight Systems.
- Error checking macro `CHECK_CUDA` is included for robustness.
- The kernel `simpleAddKernel` simply reads each element, adds a constant, and writes back.
- The main function performs two runs: one without prefetch and one with prefetch.
- The prefetch run uses `cudaMemPrefetchAsync` with `cudaCpuDeviceId` set to 0 (GPU 0). The
  page fault behavior will be visible in the Nsight Systems trace.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

// Simple kernel that adds a constant to each element
__global__ void simpleAddKernel(float* data, float val, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += val;
    }
}

// Runs the kernel with on-demand migration (no prefetch)
void runOnDemand(float* d_data, size_t N)
{
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    simpleAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, 1.0f, N);
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("On-demand migration kernel time: %f ms\n", milliseconds);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// Runs the kernel after prefetching data to the GPU
void runPrefetch(float* d_data, size_t N)
{
    // Prefetch data to GPU 0
    CHECK_CUDA(cudaMemPrefetchAsync(d_data, N * sizeof(float), 0, 0));

    // Optional: wait for prefetch to complete before kernel launch
    CHECK_CUDA(cudaDeviceSynchronize());

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    simpleAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, 1.0f, N);
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Prefetch kernel time: %f ms\n", milliseconds);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main()
{
    // Size of array: 64 MB
    const size_t N = 16 * 1024 * 1024; // 16M floats = 64 MB

    // Allocate unified memory
    float* d_data;
    CHECK_CUDA(cudaMallocManaged(&d_data, N * sizeof(float)));

    // Initialize data
    for (size_t i = 0; i < N; ++i) {
        d_data[i] = 0.0f;
    }

    // Run on-demand migration version
    runOnDemand(d_data, N);

    // Run prefetching version
    runPrefetch(d_data, N);

    // Clean up
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
```