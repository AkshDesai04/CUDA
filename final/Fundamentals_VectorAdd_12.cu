```c
/* 
Implement element-wise addition for two 2D matrices. Use a 2D grid of 2D blocks. Each thread should calculate its 2D global index `(x, y)` and use it to access the matrix elements.

Thinking process:

1. We need to write a CUDA program that adds two matrices A and B element-wise and stores the result in C.
2. The program will use a 2D grid of 2D blocks: each block is defined by blockDim.x and blockDim.y, and the grid is defined by gridDim.x and gridDim.y.
3. Each thread will compute its global coordinates:
   ```
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   ```
   These correspond to the column (x) and row (y) indices in the matrices.
4. We must handle the case where the grid does not cover the entire matrix exactly. Threads that fall outside the matrix bounds will simply return.
5. For simplicity, the matrices will be stored in row-major order. The linear index for an element at (x, y) is `y * width + x`.
6. Host code will allocate memory for A, B, and C on both host and device, initialize A and B with some values, launch the kernel, copy back C, and optionally print a few values to verify correctness.
7. We'll use float type for elements; you could change to int if desired.
8. For performance we set a typical block size of 16x16, but it can be tuned.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel: element-wise addition of two 2D matrices
__global__ void add_matrices(const float *A, const float *B, float *C, int width, int height)
{
    // Compute global x (column) and y (row) indices
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    // Check bounds
    if (x < width && y < height)
    {
        int idx = y * width + x; // row-major linear index
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    // Matrix dimensions
    const int width  = 1024;   // number of columns
    const int height = 1024;   // number of rows
    const int size   = width * height;
    const size_t bytes = size * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize host matrices
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;
            h_A[idx] = (float)(idx);            // e.g., 0, 1, 2, ...
            h_B[idx] = (float)(2 * idx);        // e.g., 0, 2, 4, ...
        }
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    // Copy host data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    add_matrices<<<gridDim, blockDim>>>(d_A, d_B, d_C, width, height);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Simple verification: print a few elements
    printf("C[0] = %f\n", h_C[0]);                     // Expected 0
    printf("C[width-1] = %f\n", h_C[width-1]);         // Expected (width-1)*3
    printf("C[(height-1)*width + (width-1)] = %f\n",
           h_C[(height-1)*width + (width-1)]);          // Expected (size-1)*3

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```