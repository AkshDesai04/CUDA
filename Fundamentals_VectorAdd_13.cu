```cuda
/*
Aim: Write a kernel that performs a conditional operation. For two input vectors A and B, compute C[i] = (A[i] > B[i]) ? A[i] : B[i] (element-wise maximum).

Thinking:
- We need a simple element‑wise maximum kernel that uses a conditional (ternary) operator.
- Host code will allocate two input arrays and an output array, initialize them with some sample data,
  copy them to the device, launch the kernel, copy the result back, and verify the output.
- Use single‑precision floats for simplicity.
- The kernel uses a simple linear indexing scheme: thread index = blockIdx.x * blockDim.x + threadIdx.x.
- Bounds checking ensures that threads with idx >= N do nothing.
- Error checking after CUDA API calls is included for robustness.
- The code is self‑contained and can be compiled with nvcc, e.g.:
      nvcc -o max_kernel max_kernel.cu
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel to compute element-wise maximum using a conditional operation
__global__ void elementwise_max(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = (A[idx] > B[idx]) ? A[idx] : B[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;               // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input data with random numbers
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)(rand()) / RAND_MAX;   // random [0,1)
        h_B[i] = (float)(rand()) / RAND_MAX;   // random [0,1)
    }

    // Allocate device memory
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_max<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure kernel execution completes

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 results
    printf("Index\tA\t\tB\t\tC (max)\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("%d\t%f\t%f\t%f\n", i, h_A[i], h_B[i], h_C[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}
```