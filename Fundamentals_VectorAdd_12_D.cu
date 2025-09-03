```cuda
/* 
 * Aim of the program: Implement matrix subtraction and multiplication.
 * 
 * Thinking:
 * 1. We want to demonstrate basic linear algebra operations on the GPU using CUDA.
 * 2. The program will allocate two input matrices A and B, and compute:
 *    - Subtraction: C_sub = A - B
 *    - Multiplication: C_mul = A * B
 * 3. For simplicity and clarity, matrices are stored in row-major order
 *    using a single contiguous float array.
 * 4. CUDA kernels:
 *    - subKernel: Each thread handles one element of the result matrix.
 *      It reads the corresponding elements from A and B, subtracts them,
 *      and writes to C_sub.
 *    - mulKernel: Naive O(n^3) multiplication: each thread computes one
 *      element of C_mul by summing over the inner dimension. For larger
 *      matrices, a tiled version would be preferable, but the naive
 *      approach keeps the code short and understandable.
 * 5. Host code:
 *    - Initialize A and B with sample values (e.g., A[i] = i, B[i] = 2*i).
 *    - Allocate device memory, copy data to device.
 *    - Launch kernels with a 2D grid of 2D blocks (size 16x16).
 *    - Copy results back to host.
 *    - Print small parts of the matrices for verification.
 * 6. Error handling:
 *    - Define a macro CUDA_CHECK to wrap CUDA API calls and report errors.
 * 7. Performance:
 *    - For demonstration, timing is optional. The focus is correctness.
 * 8. Compile with nvcc: nvcc -arch=sm_35 -o matrix_ops matrix_ops.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    __FUNCTION__, __FILE__, __LINE__,               \
                    cudaGetErrorString(err));                       \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Define block size
#define BLOCK_SIZE 16

// Kernel for matrix subtraction: C = A - B
__global__ void subKernel(const float* A, const float* B, float* C,
                          int rows, int cols) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] - B[idx];
    }
}

// Kernel for matrix multiplication: C = A * B
__global__ void mulKernel(const float* A, const float* B, float* C,
                          int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row < A_rows && col < B_cols) {
        float sum = 0.0f;
        for (int k = 0; k < A_cols; ++k) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

// Helper to initialize matrix with simple pattern
void initMatrix(float* mat, int rows, int cols, bool isA) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = isA ? (float)i : (float)(2 * i);
    }
}

// Helper to print a matrix (small portion)
void printMatrix(const float* mat, int rows, int cols, const char* name) {
    printf("Matrix %s (%dx%d):\n", name, rows, cols);
    int max_rows = (rows < 5) ? rows : 5;
    int max_cols = (cols < 5) ? cols : 5;
    for (int i = 0; i < max_rows; ++i) {
        for (int j = 0; j < max_cols; ++j) {
            printf("%8.2f ", mat[i * cols + j]);
        }
        if (max_cols < cols) printf("... ");
        printf("\n");
    }
    if (max_rows < rows) printf("... \n");
    printf("\n");
}

int main() {
    // Matrix dimensions
    const int M = 512; // rows of A
    const int N = 512; // cols of A and rows of B
    const int P = 512; // cols of B

    // Host allocation
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * P * sizeof(float);
    size_t sizeC_mul = M * P * sizeof(float);
    size_t sizeC_sub = M * N * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C_mul = (float*)malloc(sizeC_mul);
    float* h_C_sub = (float*)malloc(sizeC_sub);

    // Initialize matrices
    initMatrix(h_A, M, N, true);
    initMatrix(h_B, N, P, false); // B is 2*index for variety

    // Device allocation
    float *d_A, *d_B, *d_C_mul, *d_C_sub;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C_mul, sizeC_mul));
    CUDA_CHECK(cudaMalloc((void**)&d_C_sub, sizeC_sub));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Launch subtraction kernel
    dim3 gridDim_sub((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    subKernel<<<gridDim_sub, blockDim>>>(d_A, d_B, d_C_sub, M, N);
    CUDA_CHECK(cudaGetLastError());

    // Launch multiplication kernel
    dim3 gridDim_mul((P + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    mulKernel<<<gridDim_mul, blockDim>>>(d_A, d_B, d_C_mul, M, N, P);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_C_sub, d_C_sub, sizeC_sub, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_mul, d_C_mul, sizeC_mul, cudaMemcpyDeviceToHost));

    // Print small portions for verification
    printMatrix(h_C_sub, M, N, "C_sub");
    printMatrix(h_C_mul, M, P, "C_mul");

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_mul);
    free(h_C_sub);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_mul));
    CUDA_CHECK(cudaFree(d_C_sub));

    return 0;
}
```