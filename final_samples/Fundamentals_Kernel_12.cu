/*
Store the grid and block dimensions in `dim3` variables on the host before launching the kernel. Launch the kernel using these variables, e.g., `myKernel<<<gridDim, blockDim>>>();`. This is the standard way to configure a launch.

Thinking:
I will write a simple CUDA C program that demonstrates the use of dim3 for grid and block configuration.
The kernel will perform element‑wise addition of two float vectors.  
I choose a block size of 256 threads, which is a common choice that maps well to most GPUs.  
The grid size is calculated so that all N elements are processed: gridDim = (N + blockDim.x - 1) / blockDim.x.
The host code will:
1. Allocate and initialize host arrays.
2. Allocate device memory.
3. Copy host data to device.
4. Set up dim3 variables gridDim and blockDim on the host.
5. Launch the kernel with the syntax `myKernel<<<gridDim, blockDim>>>();`.
6. Copy the result back to host and verify a few entries.
This example shows the standard way to configure a kernel launch using dim3 variables.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void myKernel(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main()
{
    const int N = 1 << 20;   // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Configure kernel launch dimensions
    dim3 blockDim(256);                      // 256 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x); // enough blocks to cover all elements

    // Launch kernel
    myKernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify a few results
    bool success = true;
    for (int i = 0; i < 10; ++i)
    {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5)
        {
            printf("Mismatch at index %d: GPU %f, CPU %f\n", i, h_c[i], expected);
            success = false;
            break;
        }
    }
    if (success)
        printf("First 10 results are correct.\n");

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
