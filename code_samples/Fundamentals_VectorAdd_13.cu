```cpp
/*
Write a kernel that performs a conditional operation. For two input vectors A and B, compute `C[i] = (A[i] > B[i]) ? A[i] : B[i]` (element‑wise maximum).

Thinking process:
1. The problem is a classic element‑wise maximum of two vectors. A straightforward CUDA implementation involves:
   - A kernel that processes each element independently.
   - Each thread will compute the index it should operate on and perform the conditional operation.
2. To keep the example self‑contained and easy to understand, I will:
   - Use floating point vectors for simplicity.
   - Allocate and initialize host vectors with random values.
   - Copy data to device memory.
   - Launch the kernel with a sufficient number of blocks/threads.
   - Copy the result back and verify by printing a few entries.
3. Error checking is added for robustness.
4. The code is fully compilable with `nvcc` and demonstrates the kernel usage in a complete program.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Kernel: compute element‑wise maximum
__global__ void maxKernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = (A[idx] > B[idx]) ? A[idx] : B[idx];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

int main() {
    const int N = 1 << 20;          // 1M elements
    const int bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host vectors with random data
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f; // [0,100)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    maxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify by printing first 10 elements
    printf("A[i]\tB[i]\tC[i] (max)\n");
    for (int i = 0; i < 10; ++i) {
        printf("%0.2f\t%0.2f\t%0.2f\n", h_A[i], h_B[i], h_C[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```