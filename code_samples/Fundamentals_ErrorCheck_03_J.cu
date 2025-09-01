/*
Refactor your vector add program to include post-launch error checking.
Thought Process:
1. The original vector addition CUDA sample likely launches the kernel without verifying if the launch succeeded. 
2. To refactor, after the kernel launch we must:
   a. Call cudaGetLastError() to catch any launch errors.
   b. Call cudaDeviceSynchronize() to ensure kernel completion and capture any runtime errors.
3. Wrap both calls in checks and print error messages using cudaGetErrorString().
4. Keep the rest of the program (memory allocation, data transfer, kernel definition) unchanged for clarity.
5. Use standard CUDA error handling macros or inline checks.
6. The program will still compute C = A + B for vectors of size N, print a few elements, and clean up.
7. The code is a complete .cu file that compiles with nvcc.
*/

#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            std::cerr << "CUDA error at " << __FILE__            \
                      << ":" << __LINE__ << " code=" << err      \
                      << " \"" << cudaGetErrorString(err) << "\"\n"; \
            exit(EXIT_FAILURE);                                 \
        }                                                        \
    } while (0)

int main() {
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Post-launch error checking
    cudaError_t err = cudaGetLastError(); // Check for launch errors
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        // Clean up and exit
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        return EXIT_FAILURE;
    }

    // Synchronize to catch runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << "\n";
        // Clean up and exit
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        return EXIT_FAILURE;
    }

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    bool success = true;
    for (int i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": "
                      << h_C[i] << " != " << expected << "\n";
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Vector addition successful. Sample results:\n";
        for (int i = 0; i < 10; ++i) {
            std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << "\n";
        }
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
