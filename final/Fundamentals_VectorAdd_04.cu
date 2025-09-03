```cuda
/*
Implement element-wise vector subtraction (`C[i] = A[i] - B[i]`).

Thinking process:
- The goal is to provide a CUDA program that performs element-wise subtraction of two input vectors A and B, storing the result in vector C.
- We will use a CUDA kernel that takes three pointers to device memory: d_A, d_B, d_C and the number of elements N.
- Each thread will compute one element of the output vector: C[tid] = A[tid] - B[tid] if tid < N.
- For simplicity, we choose a fixed number of elements (e.g., 1<<20) but also allow the user to supply a different size via a command-line argument.
- Host code will:
  1. Allocate host memory for A, B, C.
  2. Initialize A and B with deterministic values (e.g., A[i] = i, B[i] = 2*i).
  3. Allocate device memory for d_A, d_B, d_C.
  4. Copy A and B from host to device.
  5. Configure grid and block dimensions (e.g., 256 threads per block).
  6. Launch the kernel.
  7. Copy result C back to host.
  8. Verify correctness by comparing each element to the expected value.
  9. Clean up memory.
- We include a CUDA error-checking macro to handle potential runtime errors.
- The program is self-contained and can be compiled with: nvcc vector_subtraction.cu -o vector_subtraction
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel for element-wise vector subtraction
__global__ void vectorSubtract(const float* A, const float* B, float* C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] - B[idx];
    }
}

int main(int argc, char* argv[]) {
    // Default vector size
    size_t N = 1 << 20; // 1,048,576 elements

    // Optional: allow user to specify size
    if (argc > 1) {
        N = atol(argv[1]);
    }

    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);      // A[i] = i
        h_B[i] = static_cast<float>(2 * i);  // B[i] = 2*i
    }

    // Allocate device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorSubtract<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify result
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        float expected = h_A[i] - h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %zu: host %f, device %f, expected %f\n",
                    i, h_C[i], expected, expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector subtraction successful. All %zu elements match expected values.\n", N);
    } else {
        printf("Vector subtraction failed.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```