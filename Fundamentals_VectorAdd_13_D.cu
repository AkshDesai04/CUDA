/*
Use `fmaxf()` from the CUDA math library instead of the ternary operator.

Thinking:
- The goal is to demonstrate how to use the CUDA math function fmaxf to compute the maximum
  of two floating‑point numbers in a kernel, replacing the usual ternary conditional
  expression (a > b ? a : b).
- We will create a simple element‑wise maximum kernel that operates on two input arrays
  and writes the result to an output array.
- The host code will allocate device memory, copy two sample input arrays to the device,
  launch the kernel, copy back the result, and print it to verify correctness.
- Basic error checking is included for CUDA API calls.
- The kernel uses the built‑in fmaxf function from <math.h> which is part of the CUDA
  math library and operates on single‑precision floats.
- This example is intentionally minimal so that the focus remains on the usage of fmaxf
  instead of a ternary operator.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void elementwise_max(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = fmaxf(a[idx], b[idx]);   // replace ternary with fmaxf
    }
}

int checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int main(void)
{
    const int N = 10;
    float h_a[N] = {1.0f, 3.5f, 2.2f, 4.4f, 0.0f, 5.1f, 2.8f, 3.3f, 1.1f, 9.9f};
    float h_b[N] = {0.5f, 4.0f, 2.0f, 1.0f, 7.0f, 5.0f, 3.0f, 2.5f, 8.8f, 1.2f};
    float h_c[N] = {0};

    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;

    size_t size = N * sizeof(float);

    // Allocate device memory
    checkCudaError(cudaMalloc((void**)&d_a, size), "cudaMalloc d_a");
    checkCudaError(cudaMalloc((void**)&d_b, size), "cudaMalloc d_b");
    checkCudaError(cudaMalloc((void**)&d_c, size), "cudaMalloc d_c");

    // Copy inputs to device
    checkCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), "cudaMemcpy h_a to d_a");
    checkCudaError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), "cudaMemcpy h_b to d_b");

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_max<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost), "cudaMemcpy d_c to h_c");

    // Print results
    printf("Index\t a\t b\t max(a,b)\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d\t%.2f\t%.2f\t%.2f\n", i, h_a[i], h_b[i], h_c[i]);
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
