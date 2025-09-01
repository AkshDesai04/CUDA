```cuda
/*
Implement `D[i] = sqrt(A[i]*A[i] + B[i]*B[i])` (vector magnitude for 2D vectors).

Thought process:
1. The goal is to compute the Euclidean magnitude of 2‑D vectors whose components are stored
   in two separate arrays `A` and `B`.  The result is stored in array `D`.  This is a simple
   element‑wise operation, perfectly suited for CUDA parallelism.

2. Kernel design:
   - Launch one thread per element.
   - Each thread reads `A[i]` and `B[i]`, squares them, adds, then takes the square root.
   - The result is written to `D[i]`.

3. Host code:
   - Allocate host arrays for `A`, `B`, and `D`.
   - Fill `A` and `B` with sample data (e.g., sine and cosine values or random numbers).
   - Allocate device memory with `cudaMalloc`.
   - Copy host data to device.
   - Define grid and block dimensions (`blockSize = 256`).
   - Launch kernel.
   - Copy results back to host.
   - Verify a few results by printing them.
   - Free device and host memory.

4. Error handling:
   - Wrap CUDA API calls with error checks to catch failures.
   - Check kernel launch errors with `cudaGetLastError()`.

5. Build:
   Compile with `nvcc magnitude.cu -o magnitude`.

6. Note:
   The code is self‑contained, uses only CUDA runtime API, and can be extended or
   modified to read input from files or command line if needed.
*/

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// CUDA kernel to compute vector magnitudes
__global__ void magnitudeKernel(const float *A, const float *B, float *D, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a = A[idx];
        float b = B[idx];
        D[idx] = sqrtf(a * a + b * b);   // use single precision sqrtf
    }
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",   \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

int main(void)
{
    const int N = 1 << 20;          // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_D = (float *)malloc(size);
    if (!h_A || !h_B || !h_D) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays with sample data
    for (int i = 0; i < N; ++i) {
        h_A[i] = sinf((float)i * 0.01f);
        h_B[i] = cosf((float)i * 0.01f);
    }

    // Allocate device memory
    float *d_A = NULL, *d_B = NULL, *d_D = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_D, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernel
    magnitudeKernel<<<gridSize, blockSize>>>(d_A, d_B, d_D, N);
    // Check for launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("First 10 results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("D[%d] = %f\n", i, h_D[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_D));
    free(h_A);
    free(h_B);
    free(h_D);

    return 0;
}
```