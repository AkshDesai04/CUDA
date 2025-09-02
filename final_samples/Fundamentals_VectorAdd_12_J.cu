```c
/*
Aim of the program: Time the matrix addition kernel.

Thought process:
1. The program must demonstrate a CUDA kernel that performs element-wise addition of two matrices (stored as 1-D arrays for simplicity) and measure the execution time of that kernel only.
2. We will use a square matrix of size N x N, where N is defined as a constant. The total number of elements is N*N.
3. Host arrays A, B, and C will be allocated and initialized. For simplicity, A[i] = i, B[i] = 2*i, so that the result C[i] should be 3*i.
4. Device memory will be allocated for the same arrays.
5. We will copy A and B from host to device. Timing will focus solely on the kernel execution, not the data transfer.
6. The kernel will be a simple element-wise addition: C[idx] = A[idx] + B[idx], where idx = blockIdx.x * blockDim.x + threadIdx.x. We guard against out-of-bounds.
7. For timing, we use CUDA events: cudaEventRecord before kernel launch, cudaEventRecord after, then cudaEventElapsedTime to get milliseconds.
8. After the kernel, we copy the result back to host (no timing needed here) and optionally verify correctness (by checking a few elements).
9. Finally, free all allocated memory and print the elapsed time.

Key details:
- Use block size of 256 threads, grid size computed as (total_elements + blockSize - 1)/blockSize.
- Add error checking macro to catch CUDA API failures.
- Keep the code self-contained and compile with nvcc: `nvcc -o matrix_add_timing matrix_add_timing.cu`.
- The program prints the kernel execution time in milliseconds.

*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024            // Matrix dimension (NxN)
#define BLOCK_SIZE 256    // Threads per block

// Error checking macro
#define CUDA_CHECK(err) do {                                    \
    cudaError_t err__ = (err);                                  \
    if (err__ != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err__));\
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while (0)

// Kernel: element-wise matrix addition
__global__ void matAdd(const float *A, const float *B, float *C, int totalElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    const int totalElements = N * N;
    size_t size = totalElements * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < totalElements; ++i) {
        h_A[i] = (float)i;
        h_B[i] = 2.0f * (float)i;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Timing with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch kernel
    int gridSize = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaEventRecord(start, 0));
    matAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, totalElements);
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for kernel to finish
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Compute elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host (no timing)
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Optional: verify a few elements
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            errors++;
            printf("Mismatch at %d: got %f, expected %f\n", i, h_C[i], expected);
        }
    }
    if (errors == 0) {
        printf("Verification passed for first 10 elements.\n");
    } else {
        printf("Verification failed for %d elements.\n", errors);
    }

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```