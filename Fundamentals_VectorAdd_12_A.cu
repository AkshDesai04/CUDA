```cuda
/*
 * Matrices are stored in 1D memory (row-major). The 1D index is
 * `idx = global_y * width + global_x`. Implement the matrix add using this.
 *
 * --------------------------------------------------------------------
 * Thinking:
 *
 * 1. The objective is to perform element‑wise addition of two matrices
 *    stored contiguously in 1‑D arrays (row‑major order).  
 *
 * 2. The kernel must compute a global thread index in 2‑D space:
 *      global_x = blockIdx.x * blockDim.x + threadIdx.x
 *      global_y = blockIdx.y * blockDim.y + threadIdx.y
 *    Then compute the flat index `idx = global_y * width + global_x`.
 *
 * 3. Boundary checks are required because the grid may cover more
 *    threads than actual elements (e.g., when dimensions are not a
 *    multiple of block dimensions).
 *
 * 4. For demonstration, the host code will:
 *      - Allocate and initialize two matrices A and B on the host.
 *      - Allocate device memory for A, B and C.
 *      - Copy A and B to device.
 *      - Launch the kernel with an appropriate grid/block configuration.
 *      - Copy the result C back to host.
 *      - Verify the result (optional) and print a few elements.
 *
 * 5. Use simple types (float) and sizes that compile quickly.
 *
 * 6. Error handling: wrap CUDA API calls with a macro to check return
 *    codes and abort on failure.
 *
 * 7. The code is fully self‑contained and compiles with `nvcc`.
 *
 * 8. The program will print the sum of the two matrices for a small
 *    4x4 example, showing that the computation succeeded.
 * --------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro to check CUDA API calls
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",           \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel for matrix addition using 1D index mapping
__global__ void matrixAdd(const float* A, const float* B, float* C,
                          int width, int height)
{
    // Compute 2D coordinates of the thread
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute the 1D index into the flattened arrays
    int idx = global_y * width + global_x;

    // Perform bounds checking
    if (global_x < width && global_y < height)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    // Matrix dimensions
    const int width  = 4;
    const int height = 4;
    const int size   = width * height;
    const int bytes  = size * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input matrices with simple values
    for (int i = 0; i < size; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Configure kernel launch dimensions
    dim3 blockSize(16, 16); // 256 threads per block
    dim3 gridSize((width  + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    matrixAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: print the matrices
    printf("Matrix A:\n");
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%6.1f ", h_A[y * width + x]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%6.1f ", h_B[y * width + x]);
        }
        printf("\n");
    }

    printf("\nMatrix C (A + B):\n");
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%6.1f ", h_C[y * width + x]);
        }
        printf("\n");
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