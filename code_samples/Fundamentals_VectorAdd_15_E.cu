```cuda
/*
Aim of the program:
Fuse a multiply and an add:
C[i] = A[i] * scalar1;
D[i] = B[i] + scalar2;

Thinking process:
The task is to implement a CUDA kernel that performs two independent arithmetic operations
on two input arrays A and B, producing two output arrays C and D. Both operations are
simple element-wise operations: a multiplication by a scalar for C and an addition of a
scalar for D. The natural approach is to write a single kernel that reads both input
elements, performs both operations, and writes both results. This reduces kernel launch
overhead compared to launching two separate kernels and demonstrates how to fuse
operations for better performance.

Key design decisions:
1. Use float as the data type for simplicity; this could be changed to double if higher precision is required.
2. Define a simple error-checking macro `CUDA_CHECK` to wrap CUDA API calls.
3. Use a 1D grid of 1D blocks, with a reasonable block size (256 threads) to cover the array length.
4. In `main`, allocate host memory, initialize it with some pattern, copy to device, launch the fused kernel,
   copy the results back, and finally verify by printing a few values.
5. Keep the program self-contained with minimal dependencies and no external files.

This implementation is a straightforward demonstration suitable for educational purposes
and can be easily extended or adapted for more complex fusion scenarios.
*/

#include <iostream>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            std::cerr << "CUDA error in file '" << __FILE__            \
                      << "' in line " << __LINE__ << ": "              \
                      << cudaGetErrorString(err) << std::endl;         \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Fused kernel: performs C[i] = A[i] * scalar1; D[i] = B[i] + scalar2;
__global__ void fused_multiply_add(const float* A, const float* B,
                                   float* C, float* D,
                                   float scalar1, float scalar2,
                                   int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * scalar1;
        D[idx] = B[idx] + scalar2;
    }
}

int main() {
    const int N = 1024;                     // Number of elements
    const size_t bytes = N * sizeof(float);

    // Host allocations
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    float *h_D = new float[N];

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);        // Example: A[i] = i
        h_B[i] = static_cast<float>(2 * i);    // Example: B[i] = 2*i
    }

    // Scalars
    const float scalar1 = 3.0f;
    const float scalar2 = 5.0f;

    // Device allocations
    float *d_A, *d_B, *d_C, *d_D;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMalloc(&d_D, bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch fused kernel
    fused_multiply_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D,
                                                          scalar1, scalar2, N);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure kernel completion

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost));

    // Verify results by printing first 10 elements
    std::cout << "Index\tA[i]\tB[i]\tC[i] = A[i]*" << scalar1
              << "\tD[i] = B[i]+" << scalar2 << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << i << "\t" << h_A[i] << "\t" << h_B[i]
                  << "\t" << h_C[i] << "\t" << h_D[i] << std::endl;
    }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_D;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));

    return 0;
}
```