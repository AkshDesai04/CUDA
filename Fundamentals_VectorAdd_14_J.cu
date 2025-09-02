```cuda
/* 
Handle large `double` vectors with boundary checks.

Thinking:
- We need a CUDA kernel that operates on large double vectors.
- Use element-wise addition for demonstration.
- Ensure boundary checks to avoid out-of-bounds accesses.
- Allocate memory on device using cudaMalloc.
- Use cudaMemcpy for host-to-device and device-to-host transfers.
- Use proper error checking after CUDA API calls.
- Use 256 threads per block and compute number of blocks accordingly.
- Provide a simple test in main: create two vectors of size N, initialize, run kernel, copy result, verify.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Kernel that adds two double vectors element-wise
__global__ void vectorAdd(const double* A, const double* B, double* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Boundary check
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const size_t N = 1 << 28; // ~268 million elements (~2 GB per array)
    const size_t bytes = N * sizeof(double);

    // Allocate host memory
    double *h_A = (double*)malloc(bytes);
    double *h_B = (double*)malloc(bytes);
    double *h_C = (double*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
    }

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (int)((N + threadsPerBlock - 1) / threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    int errors = 0;
    for (size_t i = 0; i < 10; ++i) {
        if (h_C[i] != 3.0) {
            ++errors;
            fprintf(stderr, "Error at index %zu: %f (expected 3.0)\n", i, h_C[i]);
        }
    }
    if (errors == 0) {
        printf("Success: All checked elements are correct.\n");
    } else {
        printf("Found %d errors.\n", errors);
    }

    // Free resources
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```