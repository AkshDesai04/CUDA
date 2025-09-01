```cpp
/*
Implement vector division: `C[i] = A[i] / B[i]`. Add a check in the kernel to prevent division by zero.

Thinking:
- The goal is to perform element‑wise division of two input arrays A and B, producing output array C.
- The kernel must guard against division by zero: if B[i] == 0, set C[i] to 0 (or some sentinel). I chose 0 for simplicity.
- We'll write a small self‑contained CUDA C++ program that:
  1. Allocates host arrays for A, B, C.
  2. Initializes A and B (including some zeros in B to test the guard).
  3. Copies them to device memory.
  4. Launches a kernel that does the division with a zero check.
  5. Copies the result back to host.
  6. Optionally prints a few elements to verify correctness.
- Error checking for CUDA API calls is included for robustness.
- The program uses `float` for simplicity and can be compiled with `nvcc`. 
*/

#include <iostream>
#include <cuda_runtime.h>
#include <vector>

// Macro to check CUDA API call results
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA error in " << __FILE__ << ":"       \
                      << __LINE__ << " : " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel performing element-wise division with division-by-zero guard
__global__ void vector_divide(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float denom = B[idx];
        if (denom != 0.0f)
            C[idx] = A[idx] / denom;
        else
            C[idx] = 0.0f; // guard against division by zero
    }
}

int main()
{
    const size_t N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host vectors
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N, 0.0f);

    // Initialize host data
    for (size_t i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);
        // Insert zeros at every 100th element of B to test division-by-zero guard
        h_B[i] = (i % 100 == 0) ? 0.0f : static_cast<float>(i + 1);
    }

    // Allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vector_divide<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify a few elements
    std::cout << "Sample results (index : A/B = C):" << std::endl;
    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << i << " : " << h_A[i] << "/" << h_B[i] << " = " << h_C[i] << std::endl;
    }
    // Check element where B == 0
    size_t zeroIdx = 100; // index with zero in B
    std::cout << zeroIdx << " : " << h_A[zeroIdx] << "/" << h_B[zeroIdx]
              << " = " << h_C[zeroIdx] << " (should be 0 due to guard)" << std::endl;

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```