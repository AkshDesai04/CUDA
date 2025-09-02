```cuda
/*
Calculate the dot product of two vectors (this is a reduction, a more advanced topic, but can be done naively with atomics).

Thinking process:
1. The goal is to compute the dot product of two vectors of equal length on the GPU using CUDA.
2. A straightforward way is to let each thread compute the product of one pair of elements and then
   accumulate the partial results into a single scalar variable using an atomic addition.
3. Since atomicAdd on floating point numbers is supported in modern GPUs, we can use a float type for simplicity.
4. Steps:
   - Allocate host arrays `h_a` and `h_b` and fill them with some test data.
   - Allocate device arrays `d_a`, `d_b`, and a device scalar `d_result`.
   - Initialize `d_result` to 0.
   - Launch a kernel with enough threads to cover all elements. Each thread:
        * Calculates the global index.
        * If the index is within bounds, multiplies `d_a[idx] * d_b[idx]`.
        * Uses `atomicAdd(&d_result, product)` to safely accumulate.
   - Copy the result back to the host and verify against a CPU reference implementation.
5. Error checking is performed after each CUDA API call to ensure correctness.
6. The code is self-contained and compiles with `nvcc -arch=sm_35 -o dot_product dot_product.cu` (or higher compute capability).
7. We choose a block size of 256 threads per block, which is a common choice.
8. The program prints the GPU-computed dot product and the CPU-computed reference for comparison.

The kernel uses atomicAdd for simplicity; more efficient reduction patterns exist, but this meets the requirement.
*/

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            std::cerr << "CUDA error in " << __FILE__ << ":"     \
                      << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                             \
        }                                                        \
    } while (0)

// Kernel to compute dot product using atomic addition
__global__ void dotProductKernel(const float* a, const float* b, float* result, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float prod = a[idx] * b[idx];
        atomicAdd(result, prod);
    }
}

int main()
{
    const size_t N = 1 << 20; // 1,048,576 elements
    const size_t bytes = N * sizeof(float);

    // Allocate and initialize host arrays
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    if (!h_a || !h_b) {
        std::cerr << "Failed to allocate host memory." << std::endl;
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_result;
    CHECK_CUDA(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_result, sizeof(float)));

    // Copy host data to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Initialize result to zero
    float zero = 0.0f;
    CHECK_CUDA(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice));

    // Determine grid and block dimensions
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    dotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    float h_result = 0.0f;
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    // Compute reference result on CPU
    double cpu_result = 0.0;
    for (size_t i = 0; i < N; ++i) {
        cpu_result += static_cast<double>(h_a[i]) * static_cast<double>(h_b[i]);
    }

    // Print results
    std::cout << "GPU dot product: " << h_result << std::endl;
    std::cout << "CPU dot product : " << cpu_result << std::endl;
    std::cout << "Relative error  : " << std::abs(h_result - static_cast<float>(cpu_result)) / std::abs(cpu_result) << std::endl;

    // Clean up
    free(h_a);
    free(h_b);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_result));

    return 0;
}
```