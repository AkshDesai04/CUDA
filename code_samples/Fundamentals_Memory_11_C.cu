/*
Use an array in constant memory, e.g., `__constant__ float coefficients[16];`. Copy 16 floats from the host and use them in a kernel.

My approach:
- Declare a constant memory array `__constant__ float coefficients[16];` that is visible to both host and device.
- On the host, create an array `h_coeffs[16]` with some sample values.
- Use `cudaMemcpyToSymbol` to copy the host array into the constant memory on the device.
- Write a simple kernel that reads the constant array and uses it to compute a value per thread; e.g., each thread sums `coefficients[i] * idx`.
- Allocate device memory for an output array, launch the kernel, copy results back to the host, and print a few elements for verification.
- Include minimal error checking and a simple `CHECK` macro to catch CUDA errors.
- The code is a complete, self‑contained CUDA C program that can be compiled with `nvcc`.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call)                                                         \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Constant memory array
__constant__ float coefficients[16];

// Kernel that uses the constant memory array
__global__ void computeKernel(float *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        // Example computation: sum of coeff * idx
        for (int i = 0; i < 16; ++i) {
            sum += coefficients[i] * (float)idx;
        }
        output[idx] = sum;
    }
}

int main()
{
    const int N = 1024; // number of elements
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Host coefficients array
    float h_coeffs[16];
    for (int i = 0; i < 16; ++i) {
        h_coeffs[i] = (float)(i + 1); // 1.0, 2.0, ..., 16.0
    }

    // Copy coefficients to constant memory
    CHECK(cudaMemcpyToSymbol(coefficients, h_coeffs, 16 * sizeof(float)));

    // Allocate device memory for output
    float *d_output = nullptr;
    CHECK(cudaMalloc((void**)&d_output, N * sizeof(float)));

    // Launch kernel
    computeKernel<<<blocks, threadsPerBlock>>>(d_output, N);
    CHECK(cudaGetLastError()); // Check kernel launch

    // Copy results back to host
    float h_output[N];
    CHECK(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print first 10 results
    printf("First 10 results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("output[%d] = %f\n", i, h_output[i]);
    }

    // Clean up
    CHECK(cudaFree(d_output));

    return 0;
}
