```c
/*
Aim of the program:
Implement vector addition for `long long int`.

Thinking and approach:
1. We need to add two vectors A and B of type `long long int` and store the result in vector C.
2. CUDA requires us to allocate memory on the device for each vector, copy the data from host to device, launch a kernel to perform element-wise addition, and copy the result back to the host.
3. Because the data type is `long long int`, we must ensure the kernel uses the same type and that memory allocations are sized accordingly (`N * sizeof(long long int)`).
4. For simplicity, we’ll use a fixed vector size (e.g., N = 10) and fill the input vectors with sample data (e.g., A[i] = i, B[i] = 2*i). This makes the program self-contained and easy to test.
5. The kernel will compute the global thread index and, if it is within bounds, perform `C[idx] = A[idx] + B[idx]`.
6. Error checking after CUDA API calls and kernel launch will help detect issues.
7. Finally, we’ll print the resulting vector to verify correctness.

This design keeps the program straightforward, demonstrates CUDA vector addition with a 64-bit integer type, and is easy to compile and run with `nvcc`.*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 10           // Size of the vectors
#define BLOCK_SIZE 256 // Threads per block

// CUDA kernel for vector addition of long long int
__global__ void vectorAdd(const long long int *A, const long long int *B, long long int *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Utility macro for checking CUDA errors
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",   \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),         \
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void) {
    long long int *h_A = NULL;
    long long int *h_B = NULL;
    long long int *h_C = NULL;

    long long int *d_A = NULL;
    long long int *d_B = NULL;
    long long int *d_C = NULL;

    size_t size = N * sizeof(long long int);

    // Allocate host memory
    h_A = (long long int *)malloc(size);
    h_B = (long long int *)malloc(size);
    h_C = (long long int *)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = (long long int)i;          // A[i] = i
        h_B[i] = (long long int)(2 * i);    // B[i] = 2*i
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorAdd<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Print the result
    printf("Vector addition results (long long int):\n");
    for (int i = 0; i < N; ++i) {
        printf("C[%d] = %lld + %lld = %lld\n", i, h_A[i], h_B[i], h_C[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```