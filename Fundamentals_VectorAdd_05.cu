/*
Implement element‑wise vector multiplication, also known as the Hadamard product (`C[i] = A[i] * B[i]`).

Thinking process:
1. The goal is to multiply two vectors element by element on the GPU.
2. I will write a CUDA kernel that each thread handles one element: `C[i] = A[i] * B[i]`.
3. On the host side, I will allocate host arrays, initialize them with sample data, and copy them to device memory.
4. After launching the kernel, I will copy the result back to host memory and verify correctness by printing a few elements.
5. I will use a simple configuration: block size of 256 threads and enough blocks to cover the array size.
6. All memory allocations will be freed at the end, and I will check for CUDA errors for safety.
7. The program is self‑contained and can be compiled with `nvcc -o hadamard hadamard.cu`.
8. I include minimal error checking for clarity but ensure the code demonstrates the core functionality of element‑wise multiplication.
*/

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define N (1 << 20)          // Vector size: 1M elements
#define BLOCK_SIZE 256       // Threads per block

// CUDA kernel for element‑wise multiplication (Hadamard product)
__global__ void hadamard(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

int main() {
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays with random values
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX; // [0,1)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX; // [0,1)
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Determine grid dimensions
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel
    hadamard<<<grid, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 elements
    printf("First 10 elements of result vector C:\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}
