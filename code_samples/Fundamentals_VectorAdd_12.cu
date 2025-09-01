```c
/*
Implement element-wise addition for two 2D matrices. Use a 2D grid of 
2D blocks. Each thread should calculate its 2D global index `(x, y)` 
and use it to access the matrix elements.

Thought process:

1. Use a CUDA kernel that computes the global (x, y) coordinate of a
   thread by adding the block offset (blockIdx * blockDim) to the
   thread offset (threadIdx).  
2. Convert the 2D coordinates to a linear index for accessing the
   1‑D arrays that represent the 2‑D matrices in row-major order.
3. Guard against out‑of‑bounds indices by checking that `x < width`
   and `y < height` before performing the addition.
4. On the host, allocate host and device memory, initialize the input
   matrices with some test values, copy them to the device, launch the
   kernel with a 2‑D grid of 2‑D blocks, copy the result back, and
   optionally verify the first few results.
5. Include error checking for all CUDA API calls using a helper macro.
6. Use a reasonably sized block (e.g., 16×16) and compute the grid
   dimensions so that all matrix elements are covered.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

__global__ void matrixAdd(const float *A, const float *B, float *C,
                          int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // global x index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // global y index

    if (x < width && y < height) {
        int idx = y * width + x; // row-major linear index
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    const int width  = 512; // number of columns
    const int height = 512; // number of rows
    const size_t bytes = width * height * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host matrices
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            h_A[idx] = (float)(idx);
            h_B[idx] = 2.0f * (float)(idx);
        }
    }

    // Allocate device memory
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy input matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(16, 16); // 16×16 threads per block
    dim3 gridDim((width  + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    matrixAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, width, height);
    CUDA_CHECK(cudaGetLastError());          // Check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());     // Wait for kernel to finish

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Sample results:\n");
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 5; ++x) {
            int idx = y * width + x;
            printf("%6.1f ", h_C[idx]);
        }
        printf("\n");
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    printf("Matrix addition completed successfully.\n");
    return 0;
}
```