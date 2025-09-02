/*
Aim: Implement element-wise multiplication: c[i] = a[i] * b[i].

Thinking:
- Use a CUDA kernel to perform the element-wise multiplication in parallel.
- Choose a reasonable default array size (e.g., 1<<20 elements) but allow the user to override via a command-line argument.
- Allocate host memory for input arrays `a` and `b`, and the output array `c`. Initialize `a` and `b` with deterministic values so that the result can be verified easily.
- Allocate device memory for `a`, `b`, and `c`. Transfer the host arrays to the device using `cudaMemcpy`.
- Define a CUDA kernel `vectorMul` that each thread computes `c[i] = a[i] * b[i]` if `i` is within bounds.
- Launch the kernel with an appropriate grid and block configuration (e.g., 256 threads per block).
- Copy the result back from device to host.
- Print a few sample results to confirm correctness.
- Add a simple error-checking macro to simplify CUDA API error handling.
- Ensure the code compiles as a .cu file and uses only CUDA runtime API calls.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Simple CUDA error-checking macro
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// CUDA kernel for element-wise multiplication
__global__ void vectorMul(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

int main(int argc, char* argv[]) {
    // Determine array size
    int N = 1 << 20; // default 1M elements
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Invalid size %d. Using default %d.\n", N, 1 << 20);
            N = 1 << 20;
        }
    }

    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i + 1);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_c, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 results
    printf("First 10 results of c[i] = a[i] * b[i]:\n");
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("c[%d] = %f\n", i, h_c[i]);
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}
