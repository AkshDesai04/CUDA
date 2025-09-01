/*
Perform the comparison for a more compute-intensive kernel, like `sin(cos(x))`. Does the GPU speedup increase?

My approach is to write a simple CUDA C program that:
1. Generates a large array of floating point numbers as input.
2. Computes sin(cos(x)) on the host (CPU) using a standard for-loop.
3. Copies the input to device memory, launches a CUDA kernel that performs sinf(cosf(x)) on each element, then copies the result back.
4. Measures execution time for both CPU and GPU paths, using std::chrono for the CPU and CUDA events for the GPU.
5. Prints out the elapsed times and the computed speedup.

Key design decisions:
- Use single-precision floats (float) to keep memory usage reasonable and to allow use of the fast `sinf`/`cosf` device functions.
- Allocate a large array (e.g., 1<<24 elements ≈ 16M) to give the GPU enough work and to amortize launch overhead.
- Use a straightforward 1D grid and block configuration (256 threads per block) which is usually a good default.
- Use `cudaEventRecord`/`cudaEventElapsedTime` for accurate GPU timing.
- Include error checking after CUDA API calls to catch any failures early.
- Provide a simple correctness check by comparing a few elements from CPU and GPU outputs.
- Keep the program self-contained so it can be compiled with `nvcc` directly.

The program will output CPU time, GPU time, and the speedup factor, allowing us to observe whether the compute-intensive sin(cos(x)) kernel yields a greater speedup than simpler kernels.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    __FUNCTION__, __FILE__, __LINE__,                 \
                    cudaGetErrorString(err));                         \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

__global__ void sin_cos_kernel(float* out, const float* in, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        out[idx] = sinf(cosf(in[idx]));
}

int main()
{
    const size_t N = 1 << 24; // 16 million elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    std::vector<float> h_in(N);
    std::vector<float> h_out_cpu(N);
    std::vector<float> h_out_gpu(N);

    // Initialize input with random values between 0 and 10
    srand(42);
    for (size_t i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    }

    // ------------------- CPU computation -------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        h_out_cpu[i] = sinf(cosf(h_in[i]));
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_elapsed = cpu_end - cpu_start;
    printf("CPU time: %.6f seconds\n", cpu_elapsed.count());

    // ------------------- GPU computation -------------------
    // Allocate device memory
    float *d_in = nullptr;
    float *d_out = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_in, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_out, bytes));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // Timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    CHECK_CUDA(cudaEventRecord(start));
    sin_cos_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));
    printf("GPU time (kernel + copy in/out): %.6f ms\n", gpu_ms);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    // Verify correctness on a few elements
    bool ok = true;
    for (size_t i = 0; i < 10; ++i) {
        float diff = fabs(h_out_cpu[i] - h_out_gpu[i]);
        if (diff > 1e-5f) {
            ok = false;
            printf("Mismatch at %zu: CPU=%.6f GPU=%.6f diff=%.6e\n",
                   i, h_out_cpu[i], h_out_gpu[i], diff);
        }
    }
    if (ok) {
        printf("Verification passed for first 10 elements.\n");
    }

    // Speedup calculation
    double gpu_seconds = gpu_ms / 1000.0;
    double speedup = cpu_elapsed.count() / gpu_seconds;
    printf("Speedup: %.2fx\n", speedup);

    return 0;
}
