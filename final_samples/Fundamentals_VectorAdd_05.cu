/*
Implement element-wise vector multiplication, also known as the Hadamard product (`C[i] = A[i] * B[i]`).

My approach is to write a minimal CUDA C program that demonstrates the Hadamard product of two float vectors.
The program performs the following steps:

1. Define a kernel `hadamard` that multiplies corresponding elements of two input arrays and stores the result in a third array.
2. In `main`:
   - Set a vector size `N` and allocate host memory for arrays `h_A`, `h_B`, and `h_C`.
   - Initialize `h_A` and `h_B` with sample data.
   - Allocate device memory for `d_A`, `d_B`, `d_C`.
   - Copy host input arrays to device.
   - Launch the kernel with a sufficient number of threads to cover all elements.
   - Copy the result back to host memory.
   - Print the input and output vectors to verify correctness.
   - Free device and host memory.
3. Include basic CUDA error checking to catch any failures in API calls.

The code is self‑contained and compiles with `nvcc`. It prints the vectors to the console for quick verification. No external dependencies other than the CUDA runtime are required.*/

#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in file '" << __FILE__ << "' at line " \
                      << __LINE__ << ": " << cudaGetErrorString(err)        \
                      << std::endl;                                         \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that performs element‑wise multiplication (Hadamard product)
__global__ void hadamard(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx];
    }
}

int main()
{
    const int N = 10;                       // Size of vectors
    const int bytes = N * sizeof(float);    // Size in bytes

    // Host allocations
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);          // 0, 1, 2, ...
        h_B[i] = static_cast<float>(2 * i);      // 0, 2, 4, ...
    }

    // Device allocations
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    hadamard<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());   // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "A\tB\tC = A*B\n";
    for (int i = 0; i < N; ++i) {
        std::cout << h_A[i] << "\t" << h_B[i] << "\t" << h_C[i] << std::endl;
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
