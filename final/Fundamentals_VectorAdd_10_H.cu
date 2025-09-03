/*
Perform the comparison for a more compute-intensive kernel, like sin(cos(x)). Does the GPU speedup increase?

The goal of this program is to compare the performance of a compute-intensive mathematical operation
on the CPU and on a CUDA-capable GPU.  We chose the operation sin(cos(x)) because it involves two
trigonometric functions and thus has a relatively high computational cost per element compared
to simple arithmetic.  By measuring the time taken to apply this operation to a large array on
both devices, we can determine whether the GPU provides a greater speedup than for less
computationally heavy kernels (e.g., a single addition or multiplication).

Approach:
1. Allocate a large array of double-precision floating-point numbers on the host.
2. Initialize the array with some values (e.g., a linear ramp).
3. Perform the operation sin(cos(x)) on the host in a simple loop, timing the execution
   with std::chrono.
4. Copy the same array to the device, launch a CUDA kernel that applies sin(cos(x)) to each
   element, and copy the results back.  We time only the kernel execution using
   cudaEvent_t for accurate measurement, excluding memory transfer times.
5. Compare the CPU time, GPU kernel time, and compute the speedup.
6. Print out the results.

Notes:
- Double-precision operations on the GPU can be slower than single precision; however,
  the kernel still benefits from parallelism.
- The size of the array is chosen to be large enough (e.g., 100 million elements) to
  amortize launch overhead and make the computation dominated by arithmetic.
- Error checking for CUDA API calls is performed using a macro to keep the code concise.
*/

#include <cstdio>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel that computes sin(cos(x)) for each element
__global__ void sincos_kernel(double* d_out, const double* d_in, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double val = d_in[idx];
        d_out[idx] = sin(cos(val));
    }
}

int main()
{
    const int N = 100'000'000; // 100 million elements
    const size_t size_bytes = N * sizeof(double);

    // Allocate host memory
    double* h_in = (double*)malloc(size_bytes);
    double* h_out_cpu = (double*)malloc(size_bytes);
    double* h_out_gpu = (double*)malloc(size_bytes);
    if (!h_in || !h_out_cpu || !h_out_gpu) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input array with a simple pattern
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<double>(i) * 0.0001; // arbitrary values
    }

    // --------------------
    // CPU computation
    // --------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        double val = h_in[i];
        h_out_cpu[i] = sin(cos(val));
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    printf("CPU time: %f seconds\n", cpu_duration.count());

    // --------------------
    // GPU computation
    // --------------------
    double* d_in = nullptr;
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, size_bytes));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size_bytes, cudaMemcpyHostToDevice));

    // Define kernel launch parameters
    const int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    sincos_kernel<<<blocks_per_grid, threads_per_block>>>(d_out, d_in, N);

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    double gpu_seconds = gpu_ms / 1000.0;
    printf("GPU kernel time (excluding transfer): %f seconds\n", gpu_seconds);

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, size_bytes, cudaMemcpyDeviceToHost));

    // Verify results (optional)
    double max_diff = 0.0;
    for (int i = 0; i < N; ++i) {
        double diff = fabs(h_out_cpu[i] - h_out_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Maximum difference between CPU and GPU results: %e\n", max_diff);

    // Clean up
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    free(h_in);
    free(h_out_cpu);
    free(h_out_gpu);

    // Compute speedup
    double speedup = cpu_duration.count() / gpu_seconds;
    printf("Speedup (CPU/GPU): %f\n", speedup);

    return 0;
}
