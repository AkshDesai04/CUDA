/*
Implement scalar subtraction: `output[i] = input[i] - scalar_value`.

My thinking:
1. The goal is to demonstrate a simple CUDA kernel that performs element-wise scalar subtraction.
2. Use a one-dimensional grid of blocks and threads to cover all array elements.
3. The kernel will check bounds to avoid out-of-range access.
4. Host code will allocate memory, initialize input, copy to device, launch kernel, copy back, and free resources.
5. Include basic error checking for CUDA API calls.
6. Use float type for simplicity; can be changed to other types if needed.
7. Provide a small example (e.g., 1024 elements) and print first few results to verify correctness.
*/

#include <iostream>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel: output[i] = input[i] - scalar
__global__ void scalarSubtractKernel(float* output, const float* input, float scalar, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] - scalar;
    }
}

int main() {
    const int N = 1024;                // Number of elements
    const float scalar = 3.5f;         // Scalar value to subtract

    // Host allocation
    float* h_input  = new float[N];
    float* h_output = new float[N];

    // Initialize input with some values, e.g., 0.0, 1.0, 2.0, ...
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Device allocation
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc((void**)&d_input,  N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, N * sizeof(float)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel: choose 256 threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    scalarSubtractKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_input, scalar, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few results
    std::cout << "First 10 results:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "input[" << i << "] = " << h_input[i]
                  << ", output[" << i << "] = " << h_output[i] << std::endl;
    }

    // Clean up
    delete[] h_input;
    delete[] h_output;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
