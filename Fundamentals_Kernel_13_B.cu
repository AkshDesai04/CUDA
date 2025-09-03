/*
Write a kernel that takes three pointers, `a`, `b`, and `c`, and computes `c[i] = a[i] + b[i]`.

Thinking process:
- The goal is to create a CUDA kernel that performs element-wise addition of two input arrays into a third array.
- The kernel will take three pointers: const float *a, const float *b, and float *c, plus an integer N indicating the number of elements.
- Each thread will compute its global index and perform the addition if the index is within bounds.
- The host code will allocate memory on both host and device, initialize the input arrays, launch the kernel with an appropriate grid/block configuration, copy the result back, and print some elements for verification.
- Error checking will be performed after CUDA API calls and kernel launch to ensure correctness.
- The code is self-contained and can be compiled with nvcc to produce an executable that demonstrates the vector addition.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for element-wise vector addition
__global__ void vectorAdd(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// Utility function for checking CUDA errors
static inline void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    const int N = 256;                   // Number of elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    if (!h_a || !h_b || !h_c)
    {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    checkCudaError(cudaMalloc((void **)&d_a, bytes), "cudaMalloc d_a");
    checkCudaError(cudaMalloc((void **)&d_b, bytes), "cudaMalloc d_b");
    checkCudaError(cudaMalloc((void **)&d_c, bytes), "cudaMalloc d_c");

    // Copy input data from host to device
    checkCudaError(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_a");
    checkCudaError(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_b");

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy d_c");

    // Verify a few results
    printf("Sample results:\n");
    for (int i = 0; i < 5; ++i)
    {
        printf("c[%d] = %f (expected %f)\n", i, h_c[i], h_a[i] + h_b[i]);
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
