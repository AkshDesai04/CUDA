/*
Handle large vectors with boundary checks in your fused kernel.

[Thoughts]
The goal is to demonstrate a CUDA kernel that performs two arithmetic operations in a single pass over the data – a typical “fused” kernel – while correctly handling vectors that may be larger than the total number of threads that can be launched.  
To accomplish this we:

1. Use a 1‑D grid of 1‑D blocks with a configurable block size (e.g. 256).  
2. Compute a global thread index `idx = blockIdx.x * blockDim.x + threadIdx.x`.  
3. Add an explicit boundary check `if (idx < N)` to avoid out‑of‑bounds memory accesses.  
4. Perform the fused operation: `y[idx] = a * x[idx] + b * z[idx]`.  
5. In the host code, allocate device buffers, initialize host data, copy to device, launch the kernel with enough blocks to cover `N`, copy the result back, and perform a simple verification or print a few elements.  
6. Add CUDA error checking macros and simple timing with `cudaEvent_t`.  
7. Keep the code self‑contained in a single `.cu` file, no external dependencies beyond the CUDA runtime.

This pattern scales to arbitrarily large `N`, limited only by available device memory, and shows how to fuse two operations into one kernel launch while guarding against out‑of‑bounds errors.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// CUDA error checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Fused kernel: y = a * x + b * z
__global__ void fused_vector_op(const float *x, const float *z, float *y,
                                float a, float b, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = a * x[idx] + b * z[idx];
    }
}

int main(int argc, char *argv[]) {
    // Parse optional command line arguments
    size_t N = 1 << 26; // Default: 67,108,864 elements (~256 MB)
    float a = 2.0f, b = 3.0f;

    if (argc >= 2) {
        N = strtoull(argv[1], NULL, 10);
    }
    if (argc >= 3) {
        a = atof(argv[2]);
    }
    if (argc >= 4) {
        b = atof(argv[3]);
    }

    printf("Vector size N = %zu\n", N);
    printf("Scalars: a = %f, b = %f\n", a, b);

    // Host memory allocation
    float *h_x = (float *)malloc(N * sizeof(float));
    float *h_z = (float *)malloc(N * sizeof(float));
    float *h_y = (float *)malloc(N * sizeof(float));
    if (!h_x || !h_z || !h_y) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (size_t i = 0; i < N; ++i) {
        h_x[i] = (float)i * 0.001f;
        h_z[i] = (float)(N - i) * 0.002f;
    }

    // Device memory allocation
    float *d_x, *d_z, *d_y;
    CUDA_CHECK(cudaMalloc((void **)&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_z, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_y, N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch configuration
    size_t gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch fused kernel
    fused_vector_op<<<gridSize, BLOCK_SIZE>>>(d_x, d_z, d_y, a, b, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Simple verification: print first 5 and last 5 results
    printf("Result sample:\n");
    for (size_t i = 0; i < 5 && i < N; ++i) {
        printf("y[%zu] = %f\n", i, h_y[i]);
    }
    if (N > 10) {
        printf("...\n");
        for (size_t i = N - 5; i < N; ++i) {
            printf("y[%zu] = %f\n", i, h_y[i]);
        }
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_y));
    free(h_x);
    free(h_z);
    free(h_y);

    return EXIT_SUCCESS;
}
