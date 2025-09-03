```cuda
/* 
Aim of the program: 
Add a boundary check: `if (global_x < width && global_y < height)` to handle matrix dimensions that are not perfect multiples of the block dimensions.

Thoughts and design:
1. The program performs element-wise addition of two matrices A and B, storing the result in matrix C.
2. To support arbitrary matrix sizes (height × width), the CUDA kernel uses a simple boundary check. Each thread calculates its global coordinates (global_x, global_y). 
   The thread only performs the addition if those coordinates are within the bounds of the matrix. 
   This guard (`if (global_x < width && global_y < height)`) ensures that for dimensions not perfectly divisible by the block size,
   the threads that fall outside the valid data region simply return without accessing memory out of bounds.
3. The kernel uses a 2D grid of 2D blocks. BLOCK_DIM is set to 16 by default, which works well on most devices.
4. The host code allocates device memory, copies input data, launches the kernel, and copies back the result.
   Basic error checking is included after each CUDA API call.
5. A simple verification step compares the computed result with a CPU reference implementation and reports success or mismatch.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 16   // Block dimension (16x16 threads)

__global__ void add_matrices(const float *A, const float *B, float *C, int height, int width)
{
    // Compute global coordinates
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check: only process if within matrix bounds
    if (global_x < width && global_y < height)
    {
        int idx = global_y * width + global_x;
        C[idx] = A[idx] + B[idx];
    }
}

// CPU reference for verification
void add_matrices_cpu(const float *A, const float *B, float *C, int height, int width)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            C[y * width + x] = A[y * width + x] + B[y * width + x];
        }
    }
}

int main()
{
    const int width  = 123;   // Example width not a multiple of BLOCK_DIM
    const int height = 77;    // Example height not a multiple of BLOCK_DIM

    const int size = width * height;
    const int bytes = size * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_C_ref = (float*)malloc(bytes);

    // Initialize input matrices with some values
    for (int i = 0; i < size; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(size - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaError_t err = cudaMalloc((void**)&d_A, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_A failed\n"); return 1; }
    err = cudaMalloc((void**)&d_B, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_B failed\n"); return 1; }
    err = cudaMalloc((void**)&d_C, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_C failed\n"); return 1; }

    // Copy inputs to device
    err = cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy d_A failed\n"); return 1; }
    err = cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy d_B failed\n"); return 1; }

    // Configure kernel launch parameters
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((width + BLOCK_DIM - 1) / BLOCK_DIM,
                 (height + BLOCK_DIM - 1) / BLOCK_DIM);

    // Launch kernel
    add_matrices<<<gridDim, blockDim>>>(d_A, d_B, d_C, height, width);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err)); return 1; }

    // Copy result back to host
    err = cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy d_C->h_C failed\n"); return 1; }

    // Compute reference result on CPU
    add_matrices_cpu(h_A, h_B, h_C_ref, height, width);

    // Verify results
    int mismatches = 0;
    for (int i = 0; i < size; ++i)
    {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-5f)
        {
            mismatches++;
            if (mismatches <= 10)
                printf("Mismatch at index %d: GPU=%f, CPU=%f\n", i, h_C[i], h_C_ref[i]);
        }
    }
    if (mismatches == 0)
        printf("Result verification PASSED.\n");
    else
        printf("Result verification FAILED with %d mismatches.\n", mismatches);

    // Clean up
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
```