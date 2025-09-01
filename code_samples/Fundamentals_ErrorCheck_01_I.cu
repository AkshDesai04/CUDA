/*
Aim of the program: Write the error checking logic in a helper function `void check_error(cudaError_t code, const char* file, int line)` and have the macro call this function.
The program demonstrates a typical CUDA helper that centralizes error checking into a single function, making the code more readable and less repetitive. The macro `CUDA_CHECK` expands to a call to `check_error` with the current file name and line number, so any CUDA API call wrapped by the macro will automatically provide contextual information on failure. The helper prints a human‑readable error message and aborts the program if a CUDA error occurs. A simple vector addition kernel is provided to illustrate how the macro is used in practice. 
*/

// Compile with: nvcc -o error_checking error_checking.cu
// Run: ./error_checking

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Helper function that checks CUDA error codes and prints context information
void check_error(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s:%d)\n",
                cudaGetErrorString(code), file, line);
        // Optionally, you can dump device properties or more details here
        exit(EXIT_FAILURE);
    }
}

// Macro that wraps CUDA calls and forwards file and line information
#define CUDA_CHECK(call) check_error((call), __FILE__, __LINE__)

// A simple kernel that adds two vectors
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20;               // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f + %f = %f\n", i, h_A[i], h_B[i], h_C[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Vector addition completed successfully.\n");
    return EXIT_SUCCESS;
}
