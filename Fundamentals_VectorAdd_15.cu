```cuda
/* 
Implement a 'fused' kernel that performs two operations at once. 
For example, `C[i] = A[i] + B[i]` and `D[i] = A[i] - B[i]`. 
This can improve performance by increasing arithmetic intensity. 

Thinking:
- The goal is to show how two arithmetic operations can be fused into a single kernel launch to reduce kernel launch overhead and increase arithmetic intensity. 
- I will write a CUDA kernel that reads A[i] and B[i] once from global memory, then writes two results, C[i] and D[i], to global memory. 
- I will use a simple 1D grid of blocks to cover all elements. 
- I will add a CUDA error checking macro to catch any runtime errors. 
- The host code will allocate host arrays, initialize them with some values, copy them to the device, invoke the kernel, copy back the results, and print a few samples to verify correctness. 
- The code is self-contained and can be compiled with nvcc. 
*/

#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error in " << __FILE__ << ":"      \
                      << __LINE__ << " : " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Fused kernel: compute C = A + B and D = A - B
__global__ void fused_add_sub(const float* A, const float* B, float* C, float* D, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a = A[idx];
        float b = B[idx];
        C[idx] = a + b;
        D[idx] = a - b;
    }
}

int main()
{
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);

    // Host allocations
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    float *h_D = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Device allocations
    float *d_A, *d_B, *d_C, *d_D;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_D, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    fused_add_sub<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N);
    CUDA_CHECK(cudaGetLastError()); // Check kernel launch

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    std::cout << "Sample results:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "A[" << i << "] = " << h_A[i]
                  << ", B[" << i << "] = " << h_B[i]
                  << ", C[" << i << "] = " << h_C[i]
                  << ", D[" << i << "] = " << h_D[i] << std::endl;
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