/*
Time the matrix addition kernel.

Thinking:
- The goal is to measure the execution time of a simple matrix addition kernel on the GPU using CUDA.
- We need to allocate two input matrices A and B and an output matrix C on both host and device.
- Fill A and B with some deterministic or random values to ensure reproducible timing; using a simple pattern like A[i] = i, B[i] = 2*i.
- Use cudaEvent_t to time only the kernel launch on the device: record a start event, launch the kernel, record a stop event, and compute the elapsed time with cudaEventElapsedTime.
- Use cudaMalloc, cudaMemcpy, and cudaMemcpyAsync if desired; for simplicity we'll use synchronous copies.
- Provide a small error-checking macro to handle CUDA API failures.
- The matrix size can be defined by a compile-time constant or taken from the command line. For flexibility, we parse an optional argument.
- Use a simple 1D grid/block arrangement: each thread processes one element; block size of 256 or 512 threads.
- After the kernel execution, we can optionally verify the result or just free resources.
- The program will print the matrix size, the elapsed GPU time in milliseconds, and optionally the host time for comparison.
- Ensure that all CUDA calls are wrapped with the error-checking macro for clarity.
- Compile with: nvcc -o matrix_add_timing matrix_add_timing.cu
- Run with optional argument: ./matrix_add_timing 1024 (for 1024x1024 matrix)
*/

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Matrix addition kernel
__global__ void matAdd(const float* A, const float* B, float* C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char* argv[]) {
    // Default matrix size (e.g., 1024x1024)
    size_t dim = 1024;
    if (argc >= 2) {
        dim = std::stoul(argv[1]);
    }
    size_t N = dim * dim; // Total elements

    // Allocate host memory
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));

    if (!h_A || !h_B || !h_C) {
        std::cerr << "Host memory allocation failed." << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    const int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    matAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Compute elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host (optional)
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Output timing result
    std::cout << "Matrix size: " << dim << "x" << dim << std::endl;
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Optional result verification (just check a few elements)
    bool correct = true;
    for (int i = 0; i < 10; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "Result verification: " << (correct ? "PASS" : "FAIL") << std::endl;

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
