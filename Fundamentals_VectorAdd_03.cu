/*
Modify your large vector addition program to handle input sizes that are not an exact multiple of the block size (e.g., 1,000,000 elements). Add a boundary check inside the kernel (if (index < N) { ... }) to prevent threads from accessing memory out of bounds.

Thinking:
- The requirement is to adjust the classic CUDA vector addition so that it works for any size N, even when N is not divisible by the block size.
- The solution uses the standard calculation for grid size: gridSize = (N + blockSize - 1) / blockSize, ensuring enough blocks to cover all elements.
- Inside the kernel, each thread calculates its global index. We then guard the memory accesses with an if (index < N) condition so that threads belonging to the padded grid that do not map to an actual element simply skip the addition.
- The host code allocates device memory for vectors A, B, and C, initializes the host vectors, copies them to the device, launches the kernel with the computed grid and block dimensions, copies the result back, and optionally verifies a few elements.
- Error checking macros are included for readability and robustness.
- The program is self‑contained and can be compiled with `nvcc -o vecadd vecadd.cu`.
*/

#include <iostream>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in file '" << __FILE__                    \
                      << "' in line " << __LINE__ << ": "                    \
                      << cudaGetErrorString(err) << " (" << err << ")\n";   \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel for vector addition with boundary check
__global__ void vectorAdd(const float *A, const float *B, float *C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    const size_t N = 1000000;        // Number of elements (not a multiple of block size)
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: print first and last few elements
    std::cout << "First 5 results:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "C[" << i << "] = " << h_C[i] << "\n";
    }
    std::cout << "Last 5 results:\n";
    for (int i = N - 5; i < N; ++i) {
        std::cout << "C[" << i << "] = " << h_C[i] << "\n";
    }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
