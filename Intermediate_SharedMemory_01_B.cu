```cuda
/*
Implement the naive kernel using a 2D grid of 2D blocks, where each thread's global (x, y) index corresponds to the C(x, y) element it computes.
Thought process:
- The goal is to demonstrate a straightforward (naive) matrix multiplication kernel that maps each thread to a unique element of the result matrix C.
- The kernel will be launched with a 2D grid of 2D blocks so that the global indices (blockIdx.x, blockIdx.y) and local indices (threadIdx.x, threadIdx.y) combine to produce the (row, col) position in C.
- We will compute the dot product of the corresponding row of A and column of B for each element of C.
- Use single-precision floats for simplicity.
- Host code will allocate matrices, initialize them, transfer to device, launch the kernel, transfer the result back, and optionally verify correctness for small matrices.
- Error checking will be performed after CUDA calls.
- Block size of 16x16 is chosen as a common value that works well on many GPUs.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    {                                                                        \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    }

// Naive matrix multiplication kernel
__global__ void matMulNaive(const float *A, const float *B, float *C, int N)
{
    // Compute global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y; // y dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x; // x dimension

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Simple function to fill a matrix with random data
void initMatrix(float *mat, int N)
{
    for (int i = 0; i < N * N; ++i) {
        mat[i] = (float)rand() / (float)RAND_MAX;
    }
}

// Optional: Verify the result against CPU computation
void verify(const float *A, const float *B, const float *C, int N)
{
    float *C_ref = (float *)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C_ref[i * N + j] = sum;
        }
    }

    int mismatches = 0;
    for (int i = 0; i < N * N; ++i) {
        if (fabs(C[i] - C_ref[i]) > 1e-3f) {
            mismatches++;
            if (mismatches <= 10) {
                printf("Mismatch at index %d: GPU=%f, CPU=%f\n",
                       i, C[i], C_ref[i]);
            }
        }
    }
    if (mismatches == 0) {
        printf("Verification passed: all elements match.\n");
    } else {
        printf("Verification failed: %d mismatches.\n", mismatches);
    }
    free(C_ref);
}

int main(int argc, char *argv[])
{
    int N = 512; // Default matrix size
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Matrix size must be positive.\n");
            return EXIT_FAILURE;
        }
    }
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    // Initialize matrices
    initMatrix(h_A, N);
    initMatrix(h_B, N);
    memset(h_C, 0, bytes);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    // Copy input matrices to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockSize(16, 16); // 256 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Optionally verify correctness for small matrices
    if (N <= 512) {
        verify(h_A, h_B, h_C, N);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Matrix multiplication of size %d x %d completed successfully.\n", N, N);
    return EXIT_SUCCESS;
}
```