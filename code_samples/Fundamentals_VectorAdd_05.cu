```cpp
/*
Aim of the program: Implement element-wise vector multiplication, also known as the Hadamard product (`C[i] = A[i] * B[i]`).

Thoughts:
1. The program will demonstrate how to compute the Hadamard product of two large vectors using CUDA.
2. A simple CUDA kernel will be defined that performs the element-wise multiplication.
3. On the host side, we will:
   - Allocate host memory for input vectors A and B, and the output vector C.
   - Initialize A and B with some values (for demonstration, we'll fill them with simple patterns).
   - Allocate device memory and copy A and B to the device.
   - Launch the kernel with an appropriate grid/block configuration.
   - Copy the result vector C back to the host.
   - Verify correctness for a small subset of elements.
   - Free all allocated resources.
4. Error checking will be performed after each CUDA API call using a helper macro `CUDA_CHECK`.
5. The program will be self-contained; compilation command (outside the comment) would be:
      nvcc -o hadamard hadamard.cu
6. To keep the example simple, the vector size is set to 1 million elements, but it can be changed via a macro or command line argument if needed.
7. Since the focus is on the element-wise product, no sophisticated optimizations (like shared memory tiling) are necessary for this example, but the kernel launch configuration will be chosen to maximize occupancy (e.g., 256 threads per block).
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define N (1 << 20) // 1 million elements
#define THREADS_PER_BLOCK 256

// Helper macro for error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel for element-wise vector multiplication
__global__ void hadamardProduct(const float *A, const float *B, float *C, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

int main(void) {
    float *h_A = NULL, *h_B = NULL, *h_C = NULL;
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    h_A = (float *)malloc(bytes);
    h_B = (float *)malloc(bytes);
    h_C = (float *)malloc(bytes);
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host vectors
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)i;            // Example pattern: 0,1,2,...
        h_B[i] = (float)(N - i);      // Example pattern: N, N-1, N-2,...
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Copy host vectors to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    size_t blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    hadamardProduct<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify results for a few elements
    bool success = true;
    for (size_t i = 0; i < 10; ++i) {
        float expected = h_A[i] * h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            fprintf(stderr, "Mismatch at index %zu: GPU=%f, CPU=%f\n", i, h_C[i], expected);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Verification passed for first 10 elements.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```