/*
Create a header file (`my_device_functions.cuh`) and put your `__device__` function in it. Include this header in your main `.cu` file.
[Thinking]
- I will create a simple CUDA program that demonstrates the use of a __device__ function defined in an external header file.
- The header file `my_device_functions.cuh` will contain a small device function, e.g., `float add(float a, float b)` that returns the sum.
- The main `.cu` file will include this header, launch a kernel that uses the device function to perform element-wise addition on two input arrays.
- Host code will allocate memory on the device, initialize input arrays, copy to device, launch kernel, copy result back, and print a few values to verify correctness.
- The code will be simple, compile with nvcc, and demonstrates separation of device functions into a header file.
*/

#include <cstdio>
#include <cuda_runtime.h>
#include "my_device_functions.cuh"

#define N 256

// Kernel that adds two arrays element-wise using the device function from the header
__global__ void addArraysKernel(const float* a, const float* b, float* c, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        c[idx] = add(a[idx], b[idx]); // call to device function
    }
}

int main()
{
    size_t bytes = N * sizeof(float);

    // Host allocations
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialize host data
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Device allocations
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Kernel launch parameters
    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    addArraysKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Print a few results
    printf("Index\tA\tB\tC = A + B\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("%d\t%.1f\t%.1f\t%.1f\n", i, h_a[i], h_b[i], h_c[i]);
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
