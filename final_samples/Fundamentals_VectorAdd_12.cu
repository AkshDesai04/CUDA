```c
/*
Aim: Implement element-wise addition for two 2D matrices. Use a 2D grid of 2D blocks. Each thread should calculate its 2D global index `(x, y)` and use it to access the matrix elements.

Thinking:
1. We need a CUDA kernel that performs element-wise addition of two matrices stored in row-major order.
2. The kernel will be launched with a 2D grid of 2D blocks. Each thread computes its global (x, y) coordinates based on blockIdx, blockDim, and threadIdx.
3. Using these coordinates we compute the linear index into the 1D flattened array: idx = y * width + x.
4. We must check bounds (x < width && y < height) to avoid out-of-bounds accesses.
5. On the host side, we allocate host matrices A, B, and C, fill A and B with sample data, copy them to the device, launch the kernel, copy result back, and verify or print a few elements.
6. We add error-checking wrappers for CUDA API calls to make debugging easier.
7. The code will compile as a .cu file. For demonstration, we use float arrays and a modest matrix size (e.g., 8x8) but the code works for any dimensions.
8. We include necessary headers, define error-checking macro, implement kernel, and the main function with memory allocations, data initialization, kernel launch configuration, and result validation.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* CUDA error checking macro */
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",   \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),        \
                    cudaGetErrorString(err));                             \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* Kernel for element-wise addition of two 2D matrices */
__global__ void addKernel(const float *A, const float *B, float *C,
                          int width, int height)
{
    // Compute global 2D indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (x < width && y < height) {
        int idx = y * width + x;
        C[idx] = A[idx] + B[idx];
    }
}

/* Utility to initialize a matrix with deterministic values */
void initMatrix(float *mat, int width, int height, float base)
{
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            mat[y * width + x] = base + y * width + x;
        }
    }
}

/* Main function */
int main(void)
{
    /* Matrix dimensions */
    const int width  = 8;   // number of columns
    const int height = 8;   // number of rows
    const int size   = width * height;
    const int bytes  = size * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host matrices.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host matrices */
    initMatrix(h_A, width, height, 1.0f);  // A: 1.0 + index
    initMatrix(h_B, width, height, 2.0f);  // B: 2.0 + index

    /* Allocate device memory */
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    /* Copy data from host to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Kernel launch configuration */
    dim3 blockDim(4, 4);          // 4x4 threads per block
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    /* Launch the kernel */
    addKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, width, height);

    /* Check for kernel launch errors */
    CHECK_CUDA(cudaGetLastError());

    /* Wait for GPU to finish */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Simple verification: print the resulting matrix */
    printf("Result matrix C (width=%d, height=%d):\n", width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%6.1f ", h_C[y * width + x]);
        }
        printf("\n");
    }

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return EXIT_SUCCESS;
}
```