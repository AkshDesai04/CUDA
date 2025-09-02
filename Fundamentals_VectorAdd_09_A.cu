```cuda
/*
Aim of the program: Implement D[i] = A[i] * B[i] + C[i] where C is a third input vector.

Thinking:
1. The kernel will perform the element-wise computation for each index i.
2. Host code will allocate memory on device for vectors A, B, C, and D.
3. Host will initialize input vectors A, B, and C with sample data.
4. The kernel will be launched with enough blocks/threads to cover all elements.
5. After execution, the result vector D will be copied back to host and optionally printed.
6. Proper CUDA error checking is added to catch any runtime errors.
7. The code uses a simple fixed-size vector (e.g., 1 << 20 elements) for demonstration.
8. The program is self-contained and can be compiled with nvcc.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 20) // Number of elements
#define THREADS_PER_BLOCK 256

// CUDA kernel for element-wise operation D[i] = A[i] * B[i] + C[i]
__global__ void elementwise_mul_add(const float* A, const float* B, const float* C, float* D, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        D[idx] = A[idx] * B[idx] + C[idx];
    }
}

// Helper function for CUDA error checking
inline void checkCuda(cudaError_t result, char const * const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(val) checkCuda((val), #val, __FILE__, __LINE__)

int main(void) {
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));
    float *h_D = (float*)malloc(N * sizeof(float));

    // Initialize host input vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.001f;
        h_B[i] = static_cast<float>(N - i) * 0.002f;
        h_C[i] = static_cast<float>(i % 100) * 0.003f;
    }

    // Device pointers
    float *d_A, *d_B, *d_C, *d_D;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_D, N * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    elementwise_mul_add<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, d_D, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_D, d_D, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Optional: print first 10 results
    printf("First 10 results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("D[%d] = %f\n", i, h_D[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
```