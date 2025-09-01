/*
For the CPU timing, use `std::chrono::high_resolution_clock`.

The aim of this program is to illustrate how to measure the execution time of a CPU routine
using `std::chrono::high_resolution_clock` and compare it with the execution time of a
CUDA kernel. The program performs a simple element‑wise vector addition on both the
CPU and the GPU. On the GPU side, CUDA events are used to measure the time taken by
the kernel itself, while on the CPU side we use the high‑resolution clock to time
the pure CPU computation.

Key points of the design:

1. **Vector size** – a large enough vector (e.g., 1<<20 elements) to make the timing
   measurable and to give the GPU a chance to show its advantage.

2. **CPU timing** – we use `std::chrono::high_resolution_clock::now()` to capture
   start and end times, and compute the difference in milliseconds.

3. **GPU timing** – CUDA events (`cudaEvent_t`) are used because they provide
   accurate timing of device operations, including kernel execution.
   The timing does not include host‑to‑device or device‑to‑host transfers,
   which are timed separately if desired.

4. **Error checking** – a simple macro (`CHECK_CUDA_ERROR`) is used to
   simplify CUDA error handling.

5. **Output** – the program prints the CPU and GPU times in milliseconds
   and verifies that the GPU result matches the CPU result.

The code is fully self‑contained and can be compiled with `nvcc`:
    nvcc -o vector_add vector_add.cu
*/

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CHECK_CUDA_ERROR(call)                                            \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__\
                      << " code=" << static_cast<int>(err)               \
                      << " \"" << cudaGetErrorString(err) << "\"\n";    \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// CUDA kernel for element‑wise vector addition
__global__ void vecAddKernel(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// CPU implementation of vector addition
void vecAddCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1 << 20;            // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C_cpu = new float[N];
    float* h_C_gpu = new float[N];

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // --------------------
    // CPU timing
    // --------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vecAddCPU(h_A, h_B, h_C_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

    // --------------------
    // GPU timing
    // --------------------
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, size));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));

    // Wait for kernel to finish
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // Get elapsed time
    float gpu_time_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_ms, start, stop));

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));

    // Clean up device memory and events
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-5) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": CPU=" << h_C_cpu[i]
                      << " GPU=" << h_C_gpu[i] << "\n";
            break;
        }
    }

    if (correct) {
        std::cout << "Results match.\n";
    } else {
        std::cout << "Results do NOT match!\n";
    }

    // Print timings
    std::cout << "CPU time:   " << cpu_duration.count() << " ms\n";
    std::cout << "GPU time:   " << gpu_time_ms << " ms\n";

    // Clean up host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;

    return 0;
}
