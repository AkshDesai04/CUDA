/*
Define the block dimensions using a `dim3` variable: `dim3 block_dim(8, 8);`
This program demonstrates how to set block dimensions in CUDA using a dim3 variable. It performs a simple vector addition of two arrays on the GPU. The block dimensions are specified as 8x8 threads per block, which results in 64 threads per block. The grid dimensions are computed based on the total number of elements to process, ensuring full coverage of the input arrays. The code includes host allocation, device memory management, kernel launch, result verification, and cleanup. It also prints a few sample output values to confirm correctness.
*/

#include <stdio.h>
#include <cuda.h>

// Kernel that adds two vectors
__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    const int N = 1024 * 1024;            // Number of elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = i * 1.0f;
        h_b[i] = (N - i) * 1.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block_dim(8, 8);          // 8x8 threads per block = 64 threads
    int threads_per_block = block_dim.x * block_dim.y;
    int grid_dim = (N + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    vectorAdd<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify a few results
    printf("Sample results:\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("h_c[%d] = %f (expected %f)\n", i, h_c[i], h_a[i] + h_b[i]);
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
