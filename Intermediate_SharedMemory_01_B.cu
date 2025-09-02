/*
Aim: Implement the naive kernel using a 2D grid of 2D blocks, where each thread's global (x, y) index corresponds to the C(x, y) element it computes.

Thinking: The problem requires a straightforward CUDA implementation of matrix multiplication (C = A × B) where each thread is responsible for computing a single element of the result matrix C. We use a 2D grid of 2D blocks: the block dimension is defined by BLOCK_SIZE in both x and y directions, and the grid dimension is the ceiling of the matrix dimensions divided by BLOCK_SIZE. Inside the kernel, each thread calculates its global row and column indices from block and thread indices. If these indices are within the bounds of the output matrix, the thread loops over the shared dimension K to accumulate the dot product for that element. The kernel uses only global memory access and a simple loop over K, which is why it is considered “naive.” We include error checking, host-side allocation and initialization, and a simple verification for small matrices. The code is fully self‑contained and can be compiled with nvcc and executed on any CUDA‑capable GPU. The block size is set to 16 by default, but can be modified. The matrices are square for simplicity, but the code can be adapted for non‑square matrices by adjusting the loop bounds and memory indexing. This implementation demonstrates the basic mapping of thread indices to matrix elements in CUDA.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Naive matrix multiplication kernel: each thread computes one C element
__global__ void naiveMatMul(const float *A, const float *B, float *C,
                            int M, int K, int N) {
    // Compute global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        // Perform dot product of row of A and column of B
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Simple host function to verify result for small matrices
void verifyResult(const float *A, const float *B, const float *C,
                  int M, int K, int N) {
    // Compute reference on host
    float *C_ref = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C_ref[i * N + j] = sum;
        }
    }
    // Compare
    int errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (fabs(C_ref[i] - C[i]) > 1e-4f) {
            if (errors < 10) {
                printf("Mismatch at index %d: GPU %f, CPU %f\n",
                       i, C[i], C_ref[i]);
            }
            ++errors;
        }
    }
    if (errors == 0) {
        printf("Verification PASSED: All elements match.\n");
    } else {
        printf("Verification FAILED: %d mismatches found.\n", errors);
    }
    free(C_ref);
}

int main(int argc, char *argv[]) {
    // Matrix dimensions
    int M = 1024; // number of rows in A and C
    int K = 1024; // number of columns in A and rows in B
    int N = 1024; // number of columns in B and C

    // Allow overriding via command line
    if (argc >= 4) {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
    }

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    // Initialize matrices with simple values
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = (float)(i % 100) / 100.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = (float)((i + 1) % 100) / 100.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeA));
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeB));
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeC));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    naiveMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Verify result for small sizes
    if (M <= 32 && N <= 32 && K <= 32) {
        verifyResult(h_A, h_B, h_C, M, K, N);
    } else {
        printf("Result computed for %d x %d matrix.\n", M, N);
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
