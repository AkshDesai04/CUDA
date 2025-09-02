```cpp
/*
Use `cudaMemPrefetchAsync` to proactively migrate data to a specific processor (CPU or a specific GPU) before it's accessed to hide migration latency.

The program demonstrates how to use CUDA Unified Memory and `cudaMemPrefetchAsync` to move data to the GPU before kernel execution and back to the CPU after the kernel finishes. The key idea is to prefetch the memory region to the device (GPU) asynchronously while the host is preparing other work, and then prefetch it back to the host asynchronously after kernel execution, allowing potential overlap of memory migration and computation. This example:

1. Allocates a large array in unified (managed) memory.
2. Uses a CUDA stream to perform asynchronous prefetch to the GPU.
3. Launches a simple kernel that squares each element.
4. Prefetches the memory back to the CPU asynchronously after the kernel.
5. Measures and prints the time spent in each phase, illustrating that migration can be hidden behind computation when possible.

The code also includes error checking and queries the device's capability to ensure unified memory is supported. It prints the device name and the total time for each step. The example is minimal yet complete and can be compiled with `nvcc` and run on any CUDA-capable GPU that supports Unified Memory (compute capability >= 6.0).
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                     \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                                  \
        }                                                                         \
    } while (0)

// Simple kernel that squares each element
__global__ void square_kernel(float *data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        data[idx] = val * val;
    }
}

int main() {
    // Determine device properties
    int device = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("Using device %d: %s\n", device, prop.name);

    // Check if Unified Memory is supported (compute capability >= 6.0)
    if (prop.major < 6) {
        fprintf(stderr, "Unified Memory requires compute capability >= 6.0. Exiting.\n");
        return EXIT_FAILURE;
    }

    const size_t N = 1 << 24; // 16M elements
    const size_t bytes = N * sizeof(float);

    // Allocate unified memory
    float *d_data;
    CHECK_CUDA(cudaMallocManaged(&d_data, bytes));

    // Initialize data on the host
    for (size_t i = 0; i < N; ++i) {
        d_data[i] = static_cast<float>(i);
    }

    // Create a CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Measure time
    auto start_total = std::chrono::high_resolution_clock::now();

    // Prefetch data to GPU before kernel launch
    auto start_prefetch_to_gpu = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemPrefetchAsync(d_data, bytes, device, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream)); // Ensure prefetch completes
    auto end_prefetch_to_gpu = std::chrono::high_resolution_clock::now();

    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    auto start_kernel = std::chrono::high_resolution_clock::now();
    square_kernel<<<gridSize, blockSize, 0, stream>>>(d_data, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream)); // Wait for kernel to finish
    auto end_kernel = std::chrono::high_resolution_clock::now();

    // Prefetch data back to CPU
    auto start_prefetch_to_cpu = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemPrefetchAsync(d_data, bytes, cudaCpuDeviceId, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream)); // Ensure prefetch completes
    auto end_prefetch_to_cpu = std::chrono::high_resolution_clock::now();

    auto end_total = std::chrono::high_resolution_clock::now();

    // Compute durations
    auto dur_prefetch_to_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_prefetch_to_gpu - start_prefetch_to_gpu).count();
    auto dur_kernel = std::chrono::duration_cast<std::chrono::milliseconds>(end_kernel - start_kernel).count();
    auto dur_prefetch_to_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_prefetch_to_cpu - start_prefetch_to_cpu).count();
    auto dur_total = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();

    printf("Prefetch to GPU: %lld ms\n", dur_prefetch_to_gpu);
    printf("Kernel execution: %lld ms\n", dur_kernel);
    printf("Prefetch to CPU: %lld ms\n", dur_prefetch_to_cpu);
    printf("Total elapsed: %lld ms\n", dur_total);

    // Simple verification (sum of squares)
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        sum += d_data[i];
    }
    printf("Sum of squared elements (first 10 printed):\n");
    for (size_t i = 0; i < 10 && i < N; ++i) {
        printf("%f ", d_data[i]);
    }
    printf("\nTotal sum: %f\n", sum);

    // Cleanup
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return 0;
}
```