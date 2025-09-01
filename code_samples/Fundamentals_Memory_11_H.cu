/*
Use the alternative syntax for `cudaMemcpyToSymbol` which takes the symbol as a variable: `cudaMemcpyToSymbol(my_const, ...)`.

I decided to create a small CUDA program that demonstrates how to use the alternative syntax of `cudaMemcpyToSymbol` where the symbol is passed as a variable. The goal is to copy a host array into constant memory on the device using the variable reference, then launch a kernel that accesses that constant memory and performs a simple operation (doubling each value). Finally, the results are copied back to host memory and printed.

The program structure is:
1. Define a constant memory array `my_const` on the device.
2. Allocate and fill a host array `h_in` with some data.
3. Use `cudaMemcpyToSymbol(my_const, h_in, size)` to copy data to the constant memory.
4. Allocate device memory for the output.
5. Launch a kernel that reads from `my_const` and writes the doubled values to the output array.
6. Copy the output back to the host and print the results.
7. Perform basic error checking and cleanup.

This demonstrates the syntax `cudaMemcpyToSymbol(my_const, ...)` and shows how to work with constant memory efficiently.
*/

#include <iostream>
#include <cuda_runtime.h>

// Constant memory array on the device
__constant__ float my_const[10];

// Simple kernel that doubles the values from constant memory
__global__ void doubleConstValues(float *out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        out[idx] = my_const[idx] * 2.0f;
    }
}

// Macro for error checking
#define CUDA_CHECK(err)                                   \
    do {                                                  \
        cudaError_t err__ = (err);                        \
        if (err__ != cudaSuccess) {                      \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err__) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                           \
        }                                                 \
    } while (0)

int main()
{
    const int N = 10;
    float h_in[N];
    float h_out[N];

    // Fill host input array
    for (int i = 0; i < N; ++i)
        h_in[i] = static_cast<float>(i);

    // Copy to constant memory using the alternative syntax
    CUDA_CHECK(cudaMemcpyToSymbol(my_const, h_in, N * sizeof(float)));

    // Allocate device memory for output
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    // Determine launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    doubleConstValues<<<blocksPerGrid, threadsPerBlock>>>(d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "Input values:" << std::endl;
    for (int i = 0; i < N; ++i)
        std::cout << h_in[i] << " ";
    std::cout << std::endl;

    std::cout << "Output values (doubled):" << std::endl;
    for (int i = 0; i < N; ++i)
        std::cout << h_out[i] << " ";
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
