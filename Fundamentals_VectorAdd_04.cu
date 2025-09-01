```cuda
/*
Aim of the program: Implement element-wise vector subtraction (C[i] = A[i] - B[i]).

Thought process and design decisions:
1. Use CUDA C to perform the subtraction on the GPU for speed and scalability.
2. Define a vector size N (e.g., 1024) that can be easily changed.
3. Allocate host memory for input vectors A and B, and output vector C.
4. Initialize A and B with sample values to demonstrate functionality.
5. Allocate device memory for A_d, B_d, and C_d.
6. Copy host input vectors to device.
7. Launch a simple CUDA kernel where each thread computes one element:
      C[i] = A[i] - B[i]
8. Use block size of 256 threads per block and compute the required number of blocks.
9. Copy the result back from device to host.
10. Print a subset of the results to verify correctness.
11. Include basic CUDA error checking macro `checkCudaErrors` for robustness.
12. Ensure all allocated memory is freed on both host and device.

The code is self-contained, compiles with `nvcc`, and demonstrates the core vector subtraction operation using CUDA.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple macro for CUDA error checking
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

// Kernel performing element-wise subtraction: C[i] = A[i] - B[i]
__global__ void vecSubKernel(const float *A, const float *B, float *C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] - B[idx];
    }
}

int main(void)
{
    const size_t N = 1024;               // Vector size
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors with sample data
    for (size_t i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);          // A[i] = i
        h_B[i] = static_cast<float>(i * 2);      // B[i] = 2*i
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void **)&d_A, bytes));
    checkCudaErrors(cudaMalloc((void **)&d_B, bytes));
    checkCudaErrors(cudaMalloc((void **)&d_C, bytes));

    // Copy inputs from host to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecSubKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());
    // Wait for GPU to finish before accessing on host
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Index\tA\tB\tC = A - B\n");
    for (size_t i = 0; i < 10; ++i)
    {
        printf("%zu\t%.1f\t%.1f\t%.1f\n", i, h_A[i], h_B[i], h_C[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    return EXIT_SUCCESS;
}
```