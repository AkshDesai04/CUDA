```cpp
/*
Implement in-place subtraction: `A[i] -= B[i]`

Thought Process:
1. The goal is to subtract each element of array B from the corresponding element of array A, modifying A directly on the GPU.
2. We'll create a simple CUDA kernel that takes pointers to A and B and the number of elements N. Each thread will handle one index `i = blockIdx.x * blockDim.x + threadIdx.x`.
3. Boundary check: if `i < N` then perform `A[i] -= B[i]`.
4. In `main`, we'll:
   - Allocate host arrays `h_A` and `h_B` of size N and initialize them with sample values.
   - Allocate device memory `d_A` and `d_B` and copy host data to device.
   - Launch the kernel with enough blocks/threads to cover N elements.
   - Copy the updated `d_A` back to host.
   - Print a few results to verify correctness.
   - Clean up device memory.
5. We'll use a simple error-checking macro to catch CUDA errors.
6. The program is self-contained and can be compiled with `nvcc` as a .cu file.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",      \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),          \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel to perform in-place subtraction: A[i] -= B[i]
__global__ void in_place_subtract(float *A, const float *B, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        A[idx] -= B[idx];
    }
}

int main() {
    const size_t N = 1024;            // Number of elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    if (!h_A || !h_B) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays with sample data
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);          // 0, 1, 2, ...
        h_B[i] = static_cast<float>(i % 10 + 0.5f); // 0.5, 1.5, ..., 9.5
    }

    // Allocate device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Determine grid and block dimensions
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    in_place_subtract<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost));

    // Print a few results to verify
    printf("First 10 results of A after in-place subtraction (A[i] -= B[i]):\n");
    for (size_t i = 0; i < 10; ++i) {
        printf("A[%zu] = %.3f\n", i, h_A[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    free(h_A);
    free(h_B);

    return 0;
}
```