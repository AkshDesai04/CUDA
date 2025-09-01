/*
Aim: Implement element-wise vector multiplication: C[i] = A[i] * B[i].
The goal of this program is to demonstrate a simple CUDA kernel that performs
element‑wise multiplication of two input vectors A and B, storing the result in
vector C. The implementation will:
1. Allocate host memory for three vectors (A, B, C) and fill A and B with test data.
2. Allocate corresponding device memory and copy A and B from host to device.
3. Launch a CUDA kernel that computes C[i] = A[i] * B[i] for all indices.
4. Copy the result vector C back to host memory.
5. Verify correctness by comparing the CUDA result with a CPU‑computed reference.
6. Clean up all allocated resources.

This program illustrates the typical CUDA workflow: memory allocation,
data transfer, kernel execution, and result validation. It also includes
basic error checking for CUDA API calls. The vector size is set to a large
value (e.g., 1,048,576 elements) to ensure a noticeable GPU workload.
The kernel uses a 1D grid of 1D blocks with a configurable block size.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

/* Kernel: element‑wise multiplication */
__global__ void vecMulKernel(const float *A, const float *B, float *C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx];
    }
}

int main(void) {
    /* Parameters */
    const size_t N = 1 << 20;      /* Number of elements (1,048,576) */
    const size_t bytes = N * sizeof(float);
    const int blockSize = 256;     /* Threads per block */

    /* Host allocations */
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_C_ref = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C || !h_C_ref) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        exit(EXIT_FAILURE);
    }

    /* Initialize input vectors with example data */
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Compute reference result on CPU */
    for (size_t i = 0; i < N; ++i) {
        h_C_ref[i] = h_A[i] * h_B[i];
    }

    /* Device allocations */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    /* Copy inputs to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Kernel launch configuration */
    int gridSize = (N + blockSize - 1) / blockSize;

    /* Launch kernel */
    vecMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verify correctness */
    int errors = 0;
    for (size_t i = 0; i < N; ++i) {
        if (h_C[i] != h_C_ref[i]) {
            if (errors < 10) { /* print first few errors */
                fprintf(stderr, "Mismatch at index %zu: GPU %f, CPU %f\n",
                        i, h_C[i], h_C_ref[i]);
            }
            ++errors;
        }
    }
    if (errors == 0) {
        printf("All results match.\n");
    } else {
        printf("Total mismatches: %d\n", errors);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}
