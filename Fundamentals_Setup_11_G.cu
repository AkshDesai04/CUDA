/*
Aim of the program: Define `dim3` variables for the grid and block sizes and use them in the kernel launch configuration.

Thinking process:
1. The user asked for a CUDA program that demonstrates how to use `dim3` variables for grid and block sizes.
2. I decided to implement a simple element‑wise addition kernel because it is short, easy to understand, and allows us to see how `dim3` is used.
3. The program will:
   - Allocate host arrays `h_A` and `h_B`.
   - Allocate device arrays `d_A`, `d_B`, and `d_C`.
   - Copy `h_A` and `h_B` to the device.
   - Define `dim3 blockSize` and `dim3 gridSize` variables.
   - Launch the kernel with the launch configuration `<<<gridSize, blockSize>>>`.
   - Copy the result back to host and verify.
4. I included basic error checking (`cudaGetLastError` and `cudaDeviceSynchronize`) to ensure the kernel launches correctly.
5. The code is self‑contained, minimal, and can be compiled with `nvcc`. It demonstrates the exact usage of `dim3` variables as requested.

*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Number of elements

// Simple element-wise addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    // Host allocations
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));

    // Initialize host data
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes using dim3
    const int blockSizeX = 256;
    dim3 blockSize(blockSizeX, 1, 1);
    dim3 gridSize((N + blockSizeX - 1) / blockSizeX, 1, 1);

    // Launch kernel with the defined grid and block configuration
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i)
    {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5)
        {
            printf("Mismatch at index %d: host %f, device %f, expected %f\n",
                   i, h_C[i], expected, h_A[i] + h_B[i]);
            success = false;
            break;
        }
    }

    if (success)
    {
        printf("Vector addition successful. All results match.\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
