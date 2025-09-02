/*
 * Aim: Add a boundary check: `if (global_x < width && global_y < height)` to handle matrix dimensions that are not perfect multiples of the block dimensions.
 *
 * Thinking:
 * The goal is to demonstrate how to safely launch a CUDA kernel on a matrix whose dimensions do not necessarily align with the chosen block size. 
 * In a CUDA kernel, each thread computes a unique global coordinate `(global_x, global_y)`. If the matrix dimensions (width, height) are not multiples of the block dimensions (blockDim.x, blockDim.y), some threads will compute indices outside the valid data range. Accessing such out‑of‑bounds memory leads to undefined behaviour.
 *
 * To avoid this, we add an explicit boundary check inside the kernel:
 * ```
 * if (global_x < width && global_y < height) {
 *     // perform operation
 * }
 * ```
 * This ensures that only threads whose coordinates lie within the actual matrix perform any memory access or computation.
 *
 * The sample program below performs element‑wise addition of two matrices. 
 * It allocates host memory, initializes two matrices with known values, copies them to the device, launches the kernel with a grid/block configuration that intentionally has a non‑perfect multiple relationship with the matrix size, and then copies the result back. 
 * Finally, it verifies the result on the host by comparing a few sample entries.
 *
 * The program is self‑contained and can be compiled with nvcc:
 *   nvcc -o matrix_add matrix_add.cu
 *   ./matrix_add
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Kernel: element‑wise addition of two matrices */
__global__ void matrixAdd(const float *A, const float *B, float *C, int width, int height)
{
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    /* Boundary check to handle non‑multiples of block dimensions */
    if (global_x < width && global_y < height)
    {
        int idx = global_y * width + global_x;
        C[idx] = A[idx] + B[idx];
    }
}

/* Helper function for CUDA error checking */
static void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    /* Matrix dimensions */
    const int width  = 1024;   /* width of matrix (columns) */
    const int height = 1025;   /* height of matrix (rows) */
    const int size   = width * height;
    const size_t bytes = size * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Initialize matrices A and B with deterministic values */
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;
            h_A[idx] = (float)(idx % 100);   /* arbitrary pattern */
            h_B[idx] = (float)((idx + 1) % 100);
        }
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc((void **)&d_A, bytes), "cudaMalloc d_A");
    checkCudaError(cudaMalloc((void **)&d_B, bytes), "cudaMalloc d_B");
    checkCudaError(cudaMalloc((void **)&d_C, bytes), "cudaMalloc d_C");

    /* Copy data from host to device */
    checkCudaError(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkCudaError(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "cudaMemcpy B");

    /* Define block and grid dimensions */
    dim3 blockSize(16, 16); /* 16x16 threads per block */
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    /* Launch kernel */
    matrixAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    /* Wait for GPU to finish */
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    /* Copy result back to host */
    checkCudaError(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C");

    /* Simple verification: compare a few elements */
    int errors = 0;
    for (int i = 0; i < 10; ++i)
    {
        int y = i % height;
        int x = i % width;
        int idx = y * width + x;
        float expected = h_A[idx] + h_B[idx];
        if (fabs(h_C[idx] - expected) > 1e-5)
        {
            fprintf(stderr, "Mismatch at (%d, %d): host=%f, device=%f\n",
                    y, x, expected, h_C[idx]);
            errors++;
        }
    }

    if (errors == 0)
        printf("Verification passed: sample elements match.\n");
    else
        printf("Verification failed: %d mismatches.\n", errors);

    /* Free resources */
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
