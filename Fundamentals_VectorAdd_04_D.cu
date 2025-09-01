/*
Aim: Implement C[i] = abs(A[i] - B[i]).

Thinking:
- The program will compute the element‑wise absolute difference between two arrays A and B
  and store the result in array C.
- We choose float data type for generality; the kernel uses `fabsf` to compute absolute values
  on the device.
- Host code will allocate and initialize input arrays, transfer them to the GPU, launch a
  CUDA kernel, copy the result back, and optionally verify a few elements.
- Error handling is provided via the `CUDA_CHECK` macro to catch CUDA runtime errors.
- The kernel is launched with a 1D grid of blocks, each block containing 256 threads.
  The number of blocks is calculated to cover all `N` elements.
- For demonstration purposes, the input arrays are initialized with simple deterministic
  values so that the output can be easily inspected.
- The program prints a small subset of results and a checksum to confirm correctness.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Error checking macro
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// Kernel to compute absolute difference
__global__ void abs_diff(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = fabsf(A[idx] - B[idx]);
    }
}

int main(void)
{
    const int N = 1 << 20;      // Number of elements (1,048,576)
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;           // 0.0f, 1.0f, 2.0f, ...
        h_B[i] = (float)(N - i);     // N, N-1, N-2, ...
    }

    // Allocate device memory
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    abs_diff<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());          // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Sample results (index: A, B, |A-B|):\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d: %f, %f, %f\n", i, h_A[i], h_B[i], h_C[i]);
    }

    // Simple checksum
    double checksum = 0.0;
    for (int i = 0; i < N; ++i) {
        checksum += h_C[i];
    }
    printf("Checksum of all results: %f\n", checksum);

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}
