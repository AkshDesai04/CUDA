```cpp
/*
 * Aim of the program: Handle large vectors with boundary checks in your fused kernel.
 *
 * My thinking process:
 * 1. The goal is to demonstrate how to process large vectors on the GPU using a single
 *    kernel that performs multiple operations (a "fused" kernel) while ensuring that
 *    all thread indices that are out of bounds do not access invalid memory.
 * 2. I'll create two input vectors A and B, a scalar alpha, and compute an output
 *    vector C such that C[i] = A[i] + alpha * B[i]. This combines an addition and
 *    a multiplication in one kernel launch.
 * 3. To handle very large vectors, the kernel will use a typical 1D grid of thread
 *    blocks. Each thread calculates a global index `idx` and checks if it is less
 *    than the vector size N before performing any memory access. This guard prevents
 *    out‑of‑bounds reads/writes when N is not an exact multiple of the total
 *    number of threads.
 * 4. The program will:
 *    - Allocate large host arrays (e.g., 10 million elements) and initialize them.
 *    - Allocate corresponding device arrays and copy the host data to device.
 *    - Launch the fused kernel with a reasonable block size (e.g., 256 threads/block).
 *    - Copy the result back to host and perform a quick sanity check.
 * 5. I will include a simple CUDA error checking macro to surface any launch or
 *    memory allocation failures.
 * 6. All memory will be freed before program exit. The code is written in pure CUDA
 *    C (CUDA extensions to C++), compiled with `nvcc -arch=sm_52 <file>.cu`.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// Macro for checking CUDA errors following a CUDA API call or kernel launch
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",\
                    __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Fused kernel: C[i] = A[i] + alpha * B[i]
__global__ void fusedKernel(const float *A, const float *B, float *C, int N, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + alpha * B[idx];
    }
}

int main(int argc, char *argv[]) {
    // Vector size (use a large number; adjust based on device memory)
    const size_t N = 10'000'000; // 10 million elements (~160 MB per array)
    const float alpha = 2.5f;

    // Allocate host memory
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)i * 0.001f;
        h_B[i] = (float)(N - i) * 0.002f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the fused kernel
    fusedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, (int)N, alpha);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for completion

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Simple sanity check: print first 5 and last 5 elements
    printf("First 5 results:\n");
    for (int i = 0; i < 5; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }
    printf("Last 5 results:\n");
    for (int i = (int)N - 5; i < (int)N; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
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