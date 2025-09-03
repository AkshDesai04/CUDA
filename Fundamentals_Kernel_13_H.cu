```cpp
/*
Aim of the program: In the host code, use `cudaMemset` to initialize the input array `a` to a specific value (e.g., 1) on the device before launching the kernel.

Thinking process:
- The goal is to demonstrate how to use `cudaMemset` for initializing device memory from the host side.
- We will allocate two device arrays: `d_a` for the input and `d_b` for the output.
- The input array `d_a` will be initialized with the value `1` for each element using `cudaMemset`.
  Note: `cudaMemset` sets memory byte-wise. Setting each byte to `1` will fill an `int` with `0x01010101` (i.e., 16843009). 
  For simplicity and clarity, this is acceptable for the demonstration.
- A simple kernel will multiply each element by `2` and write the result to `d_b`.
- After kernel execution, we copy `d_b` back to the host and print the values to verify that the initialization worked.
- We include basic error checking for CUDA calls.
- The code is self‑contained and can be compiled with `nvcc` into an executable.
*/

#include <iostream>
#include <cuda_runtime.h>

#define N 10

// Simple error checking macro
#define CHECK_CUDA(call)                                          \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error in " << __FILE__ << ":"      \
                      << __LINE__ << " : " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    }

// Kernel that doubles each element of array a and stores in array b
__global__ void doubleKernel(const int *a, int *b, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        b[idx] = a[idx] * 2;
    }
}

int main()
{
    int *d_a = nullptr;
    int *d_b = nullptr;
    int h_b[N];

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_a, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_b, N * sizeof(int)));

    // Initialize device array 'd_a' to 1 using cudaMemset
    // Each byte will be set to 1, so each int becomes 0x01010101
    CHECK_CUDA(cudaMemset(d_a, 1, N * sizeof(int)));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    doubleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);
    CHECK_CUDA(cudaGetLastError());  // Check for kernel launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "Result of doubling the initialized array:" << std::endl;
    for (int i = 0; i < N; ++i)
    {
        std::cout << "h_b[" << i << "] = " << h_b[i] << std::endl;
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    return 0;
}
```