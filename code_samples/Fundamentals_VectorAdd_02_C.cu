```cpp
/*
Aim: Wrap the grid/block calculation and kernel launch into a helper function.

In this implementation we demonstrate how to encapsulate the logic for
computing grid and block dimensions as well as launching a CUDA kernel
into a reusable helper function.  The example performs a simple vector
addition on the GPU:

    C[i] = A[i] + B[i]

The helper function `launchVectorAdd` receives device pointers to the
input and output arrays along with the number of elements.  Inside the
function we determine an appropriate block size (fixed at 256 threads
per block) and compute the necessary grid size to cover all elements.
The kernel is then launched with the calculated launch configuration.
This pattern makes the code cleaner and separates concerns: kernel
definition, device memory management, and launch configuration are
kept distinct.  Error checking macros are used throughout to ensure
that CUDA calls succeed, and the main function sets up data,
invokes the helper, copies results back, and verifies correctness.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

// Error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    \
                      << " : " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel: vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Helper function that wraps grid/block calculation and kernel launch
void launchVectorAdd(const float* d_A, const float* d_B, float* d_C, size_t N)
{
    const int BLOCK_SIZE = 256; // Threads per block
    int gridSize = static_cast<int>((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

int main()
{
    const size_t N = 1 << 20; // 1M elements
    std::vector<float> h_A(N, 1.0f), h_B(N, 2.0f), h_C(N, 0.0f);

    // Device pointers
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel via helper
    launchVectorAdd(d_A, d_B, d_C, N);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results
    for (size_t i = 0; i < N; ++i) {
        assert(h_C[i] == h_A[i] + h_B[i]);
    }
    std::cout << "Vector addition successful, all results are correct." << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```