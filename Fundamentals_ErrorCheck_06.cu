```cpp
/* 
Aim: Create a C++ helper function `void checkKernelLaunch()` that calls `cudaDeviceSynchronize()` and checks for errors. Call this function after every kernel launch in your programs to ensure the kernel executed without an asynchronous error.

Thinking:
- The goal is to provide a reusable helper that can be inserted immediately after any CUDA kernel launch.
- We need to capture both the synchronous error that might occur during the kernel launch itself (`cudaGetLastError`) and any asynchronous errors that surface during kernel execution (`cudaDeviceSynchronize`).
- The helper will check the return status of both calls, print a descriptive error message, and terminate the program if an error is detected.
- For demonstration, a simple vector addition kernel is implemented. After launching the kernel, `checkKernelLaunch()` is invoked to validate the launch.
- This file is self‑contained and can be compiled with `nvcc -o vector_add vector_add.cu`.
*/

#include <iostream>
#include <cuda_runtime.h>

// Helper function to check kernel launch success
void checkKernelLaunch()
{
    // Check for errors during kernel launch (synchronous)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    // Wait for the device to finish and check for asynchronous errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel execution error: " << cudaGetErrorString(err) << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// Simple vector addition kernel
__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main()
{
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize host data
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check kernel launch and execution
    checkKernelLaunch();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify results (simple check)
    bool success = true;
    for (int i = 0; i < N; ++i)
    {
        if (h_C[i] != h_A[i] + h_B[i])
        {
            std::cerr << "Verification failed at index " << i << std::endl;
            success = false;
            break;
        }
    }

    if (success)
        std::cout << "Vector addition successful!" << std::endl;

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    // Reset device
    cudaDeviceReset();

    return 0;
}
```