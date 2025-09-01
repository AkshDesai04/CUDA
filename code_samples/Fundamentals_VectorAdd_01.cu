```c
/*
Implement the canonical first parallel program: element-wise vector addition.
Create two input vectors (A and B) and one output vector (C) of size 256.
Initialize A and B on the host.
Launch a kernel with one block of 256 threads.
Each thread `i` should compute `C[i] = A[i] + B[i]`.

Thinking:
The problem requires writing a CUDA program that performs element-wise addition of two vectors.
The vectors are small (size 256) so we can launch a single block with 256 threads.
The host code will:
1. Allocate host arrays for A, B, and C.
2. Initialize A and B with some values (e.g., sequential numbers or random).
3. Allocate device memory for A, B, C.
4. Copy A and B from host to device.
5. Define a simple kernel that takes device pointers and the vector length.
   Each thread will compute its global index `idx` and if `idx < N` perform the addition.
6. Launch the kernel with 1 block of 256 threads (dim3 block(256), dim3 grid(1)).
7. Copy the result vector C back to host.
8. Optionally print first few results for verification.
9. Free device and host memory.
Error checking macros are added to simplify CUDA API error handling.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple macro for checking CUDA errors
#define CUDA_CHECK(err) do {                                            \
    cudaError_t err_ = (err);                                           \
    if (err_ != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(err_));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                   \
} while (0)

// Kernel: each thread computes C[i] = A[i] + B[i]
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 256;
    const int bytes = N * sizeof(float);

    // Host allocation
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;            // e.g., 0.0, 1.0, 2.0, ...
        h_B[i] = (float)(N - i);      // e.g., 256.0, 255.0, ...
    }

    // Device allocation
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel: 1 block of 256 threads
    dim3 threadsPerBlock(256);
    dim3 numBlocks(1);
    vectorAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify results (print first 10 elements)
    printf("First 10 results of C = A + B:\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```