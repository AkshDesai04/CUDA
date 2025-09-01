```cpp
/*
Aim: Time your SAXPY implementation for a large vector.

Thought process:
- SAXPY (Single-Precision A·X plus Y) is a classic linear algebra operation
  defined as y[i] = a * x[i] + y[i] for all elements of the vectors.
- To evaluate performance, we implement the kernel on the GPU and measure
  its execution time using CUDA events which provide nanosecond resolution
  for GPU activity.
- For comparison, we also run the same operation on the CPU using
  std::chrono and report both times.
- The vector size is large enough (default 16 million elements) to ensure
  we exercise the GPU fully, but the program accepts a custom size via the
  command line.
- Error checking is performed on all CUDA calls to catch problems early.
- After execution, a simple verification of a few elements is performed
  to ensure correctness of the GPU computation.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>

// CUDA error checking macro
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":"      \
                      << __LINE__ << " code=" << err << " ("      \
                      << cudaGetErrorString(err) << ")\n";         \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// SAXPY kernel: y[i] = a * x[i] + y[i]
__global__ void saxpy_kernel(const float a, const float *x, float *y, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        y[idx] = a * x[idx] + y[idx];
}

int main(int argc, char *argv[])
{
    // Parse vector size from command line or use default
    const size_t default_size = 1 << 24;  // 16,777,216 elements (~64 MB)
    size_t n = (argc > 1) ? std::stoul(argv[1]) : default_size;
    const float a = 2.5f;  // scaling factor

    std::cout << "Vector size: " << n << " elements\n";

    // Allocate host memory
    float *h_x = new float[n];
    float *h_y = new float[n];

    // Initialize host vectors
    for (size_t i = 0; i < n; ++i) {
        h_x[i] = 1.0f;          // arbitrary values
        h_y[i] = 2.0f;          // arbitrary values
    }

    // Allocate device memory
    float *d_x = nullptr;
    float *d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice));

    // Determine launch configuration
    const int threads_per_block = 256;
    const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // Timing the GPU kernel with CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    saxpy_kernel<<<blocks_per_grid, threads_per_block>>>(a, d_x, d_y, static_cast<int>(n));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU SAXPY time: " << milliseconds << " ms\n";

    // Clean up CUDA events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few elements
    bool correct = true;
    for (size_t i = 0; i < 10; ++i) {
        float expected = a * h_x[i] + 2.0f;  // original y was 2.0
        if (fabs(h_y[i] - expected) > 1e-5f) {
            correct = false;
            std::cerr << "Mismatch at index " << i
                      << ": got " << h_y[i]
                      << ", expected " << expected << "\n";
            break;
        }
    }
    if (correct)
        std::cout << "GPU result verified for first 10 elements.\n";

    // CPU baseline timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; ++i)
        h_y[i] = a * h_x[i] + h_y[i];
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = cpu_end - cpu_start;
    std::cout << "CPU SAXPY time: " << cpu_ms.count() << " ms\n";

    // Clean up
    delete[] h_x;
    delete[] h_y;
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    return 0;
}
```