/*
Compare the performance of your CUDA vector addition kernel against a simple, single-threaded C++ loop on the CPU. Time both versions for a very large vector (e.g., 10 million elements). Remember to only time the computation, not the data transfers, for a fair comparison.

Thinking:
- Use a large vector size (10 million) to get measurable times on both CPU and GPU.
- For CPU: use a simple for-loop that adds two arrays element-wise and store in a third array.
- For GPU: write a CUDA kernel that does the same addition.
- To compare fairly, we only time the computation part:
  * CPU: wrap the loop body with std::chrono::high_resolution_clock timestamps.
  * GPU: use CUDA events (cudaEventRecord) to time only the kernel execution, excluding host-to-device and device-to-host copies.
- Use cudaMalloc and cudaMemcpy for device memory, but the copies are not included in the timed section.
- Use block size 256 threads and compute grid size accordingly.
- After timing, print both times and optionally verify correctness on a few elements.
- Add error checking macro for CUDA API calls for robustness.
- The code is self-contained in a single .cu file and can be compiled with nvcc.
*/

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " "   \
                      << cudaGetErrorString(err) << std::endl;                    \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

__global__ void vectorAddKernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 10 * 1000 * 1000; // 10 million elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C_cpu = (float*)malloc(bytes);
    float* h_C_gpu = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        std::cerr << "Failed to allocate host memory." << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize input arrays with some data
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(std::sin(i));
        h_B[i] = static_cast<float>(std::cos(i));
    }

    // --------------------- CPU timing ---------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        h_C_cpu[i] = h_A[i] + h_B[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;

    // --------------------- GPU setup ---------------------
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy input data to device (not timed)
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // --------------------- GPU timing ---------------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    vectorAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for kernel to finish
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU kernel time: " << milliseconds << " ms" << std::endl;

    // Copy result back to host (not timed)
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    // --------------------- Validation ---------------------
    bool correct = true;
    for (int i = 0; i < N; i += N / 10) { // sample 10 points
        if (std::fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": CPU=" << h_C_cpu[i]
                      << " GPU=" << h_C_gpu[i] << std::endl;
            correct = false;
            break;
        }
    }
    if (correct) {
        std::cout << "Result validation passed." << std::endl;
    } else {
        std::cerr << "Result validation failed." << std::endl;
    }

    // --------------------- Cleanup ---------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
