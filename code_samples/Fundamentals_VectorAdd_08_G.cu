/*
Aim: Handle large vectors with boundary checks.

Thinking:
To demonstrate handling large vectors safely in CUDA, the program allocates two large host vectors and a result vector. It transfers them to the GPU, launches a vector addition kernel where each thread processes one element but first checks if its global index is within bounds (i.e., idx < N). This boundary check prevents out‑of‑bounds memory accesses when the number of elements exceeds the total number of launched threads. The program includes CUDA error checking macros, uses grid and block dimensions that scale with the vector size, and verifies the GPU result against the CPU computation before freeing resources. This approach illustrates safe memory access patterns for large datasets in CUDA C.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel for vector addition with boundary checks
__global__ void vecAdd(const float *A, const float *B, float *C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    const size_t N = 1 << 24; // ~16 million elements (~64MB for float)
    const size_t bytes = N * sizeof(float);

    // Allocate pinned host memory for better transfer performance
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    float *h_C_ref = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C || !h_C_ref) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors with random data
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Compute reference result on CPU
    for (size_t i = 0; i < N; ++i) {
        h_C_ref[i] = h_A[i] + h_B[i];
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    const int THREADS_PER_BLOCK = 256;
    int blocks = (int)((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // Launch kernel
    vecAdd<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %zu: GPU=%f CPU=%f\n", i, h_C[i], h_C_ref[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector addition successful. All results match.\n");
    } else {
        printf("Vector addition failed. Results do not match.\n");
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
