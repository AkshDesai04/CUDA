```cuda
/* 
A kernel that computes `C[i] = a*A[i] + b*B[i]` is a fused operation.

Thinking:
- The goal is to demonstrate a fused kernel that performs both multiplication and addition in a single pass over the data.
- The kernel will take scalars a and b, two input arrays A and B, and write the result into C.
- To keep the example self-contained, the host code will allocate and initialize sample data, copy it to the GPU, launch the kernel, and copy the results back.
- Error checking is added via a helper macro to catch CUDA API failures.
- The kernel uses a simple 1D grid/block layout, suitable for any array size.
- The host code also prints a few sample results for verification.
*/

#include <iostream>
#include <cuda_runtime.h>

// Helper macro for error checking
#define checkCuda(call)                                                      \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__     \
                      << " code=" << err << " \"" << cudaGetErrorString(err) \
                      << "\"\n";                                            \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Fused kernel: C[i] = a*A[i] + b*B[i]
__global__ void fusedAddMul(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            float a,
                            float b,
                            size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = a * A[idx] + b * B[idx];
    }
}

int main()
{
    const size_t N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.01f;
        h_B[i] = static_cast<float>(N - i) * 0.02f;
    }

    // Scalars
    float a = 2.5f;
    float b = -1.5f;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc((void**)&d_A, bytes));
    checkCuda(cudaMalloc((void**)&d_B, bytes));
    checkCuda(cudaMalloc((void**)&d_C, bytes));

    // Copy data from host to device
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    fusedAddMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, a, b, N);
    checkCuda(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: print first 5 results
    std::cout << "First 5 results of C[i] = a*A[i] + b*B[i]:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "C[" << i << "] = " << h_C[i] << '\n';
    }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));

    return 0;
}
```