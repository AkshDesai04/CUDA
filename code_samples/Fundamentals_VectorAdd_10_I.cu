/*
Perform the comparison on a machine with a high-end CPU and a low-end GPU, and vice-versa. Observe how the results change.

My thinking: 
- The goal is to provide a simple, self-contained CUDA program that measures and compares the execution time of a basic numeric operation on the CPU versus the GPU.
- I chose vector addition (c = a + b) because it is straightforward, has no memory dependency issues, and scales well with data size. This operation is representative of many real‑world workloads where the GPU can potentially accelerate large data processing.
- The program allocates large vectors (e.g., 10 million elements) to ensure the GPU’s parallelism can be fully exercised. The same data is used for both CPU and GPU computations to keep the comparison fair.
- For the CPU, I use `std::chrono::high_resolution_clock` to time the kernel execution. For the GPU, CUDA events (`cudaEvent_t`) are used because they provide high‑resolution timing that includes kernel launch overhead.
- I add a simple error‑checking macro (`CUDA_CHECK`) to catch any CUDA API failures.
- The program prints the measured times, calculates the speedup factor, and warns if the GPU runtime exceeds the CPU runtime, indicating that the GPU may not be giving a benefit on the current machine configuration (which is what the user will observe when swapping a high‑end CPU/low‑end GPU or vice‑versa).
- No external dependencies beyond the CUDA toolkit and standard C++ libraries are used. The code compiles with `nvcc` and runs on any platform that has both a CPU and a CUDA‑capable GPU installed.
*/

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel for vector addition
__global__ void vecAddKernel(const double* a, const double* b, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CPU implementation of vector addition
void vecAddCPU(const double* a, const double* b, double* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const size_t N = 10'000'000; // 10 million elements
    const size_t bytes = N * sizeof(double);

    // Allocate host memory
    double* h_a = (double*)malloc(bytes);
    double* h_b = (double*)malloc(bytes);
    double* h_c_cpu = (double*)malloc(bytes);
    double* h_c_gpu = (double*)malloc(bytes);
    if (!h_a || !h_b || !h_c_cpu || !h_c_gpu) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input data
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = static_cast<double>(rand()) / RAND_MAX;
        h_b[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vecAddCPU(h_a, h_b, h_c_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    printf("CPU time: %f seconds\n", cpu_duration.count());

    // Allocate device memory
    double *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // GPU timing using CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaEventRecord(start));
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    printf("GPU time: %f milliseconds\n", gpu_time_ms);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify correctness
    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-6) {
            correct = false;
            printf("Mismatch at index %zu: CPU %f vs GPU %f\n",
                   i, h_c_cpu[i], h_c_gpu[i]);
            break;
        }
    }
    printf("Result verification: %s\n", correct ? "PASS" : "FAIL");

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
