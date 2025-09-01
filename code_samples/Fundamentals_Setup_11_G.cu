```cuda
/*
Aim of the program:
Define `dim3` variables for the grid and block sizes and use them in the kernel launch configuration.

Thinking:
The requirement is to demonstrate the definition and usage of `dim3` variables for the grid and block sizes
in a CUDA kernel launch. I will write a simple vector addition program that:
1. Allocates two input vectors and an output vector on the host.
2. Copies the input vectors to device memory.
3. Defines `dim3` variables `blockDim` and `gridDim` to specify the size of each block and the number of blocks.
   The block size will be set to 256 threads per block (a common choice) and the grid size will be calculated
   to cover the entire input array.
4. Launches a CUDA kernel using the `<<<gridDim, blockDim>>>` syntax.
5. Copies the result back to the host and prints a few elements to verify correctness.
6. Cleans up device and host memory.

This program will compile as a `.cu` file and can be built with `nvcc`.*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Define block and grid sizes using dim3
    dim3 blockDim(256); // 256 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x); // enough blocks to cover N elements

    // Launch kernel
    vecAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("First 5 results:\n");
    for (int i = 0; i < 5; ++i)
        printf("C[%d] = %f\n", i, h_C[i]);

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```