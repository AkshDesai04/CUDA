/*
Aim: Use `fprintf(stderr, ...)` to print the error message to standard error instead of standard out.
Thinking:
- The assignment requires a CUDA program that demonstrates error handling.
- We'll write a simple vector addition kernel.
- On the host side, all error checks (memory allocation, kernel launch, synchronization) will report errors to standard error using fprintf(stderr, ...).
- This includes checking cudaMalloc, cudaMemcpy, kernel launch errors via cudaGetLastError, and cudaDeviceSynchronize.
- We'll provide a helper macro CUDA_CHECK to simplify error handling.
- The code will compile with nvcc and run on any CUDA-capable GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro to check CUDA API calls and print errors to stderr
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "[%s:%d] CUDA error in '%s': %s\n",   \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Simple vector addition kernel
__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main(void)
{
    const int N = 1 << 20;            // 1M elements
    const size_t bytes = N * sizeof(float);

    // Host allocations
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Initialize input arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        // Cleanup before exiting
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        exit(EXIT_FAILURE);
    }

    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify result
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            errors++;
            if (errors <= 10) { // Print first few errors
                fprintf(stderr, "Mismatch at index %d: host=%f, device=%f\n", i, expected, h_C[i]);
            }
        }
    }

    if (errors == 0) {
        printf("Vector addition successful. No errors found.\n");
    } else {
        fprintf(stderr, "Vector addition found %d errors.\n", errors);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
