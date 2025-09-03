```cuda
/*
Aim of the program: 
Matrices are stored in 1D memory (row-major). The 1D index is idx = global_y * width + global_x. 
Implement the matrix add using this.

Thought process:
1. The program will perform element‑wise addition of two matrices A and B into C.
2. All matrices are stored in 1D contiguous arrays in row‑major order.
3. The kernel will compute the global thread coordinates (global_x, global_y) from blockIdx/ThreadIdx.
4. The linear index is computed as idx = global_y * width + global_x.
5. Bounds checking ensures we do not write past the array.
6. Host code will:
   - Allocate host arrays, fill them with sample data.
   - Allocate device arrays and copy data.
   - Define block and grid dimensions.
   - Launch the kernel.
   - Copy result back and print a small subset for verification.
   - Free all resources.
7. Basic error checking macro is used for CUDA API calls.
8. The program is fully self‑contained and can be compiled with nvcc.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors following a CUDA API call */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Kernel to perform matrix addition in row-major order */
__global__ void matrixAddKernel(const float *A, const float *B, float *C,
                                int width, int height)
{
    /* Compute global coordinates of the thread */
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    /* Bounds check */
    if (global_x >= width || global_y >= height)
        return;

    /* Compute linear index */
    int idx = global_y * width + global_x;

    /* Perform addition */
    C[idx] = A[idx] + B[idx];
}

int main(void)
{
    /* Define matrix dimensions */
    const int width  = 1024;   /* number of columns */
    const int height = 1024;   /* number of rows */
    const int size   = width * height;
    const size_t bytes = size * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host matrices.\n");
        exit(EXIT_FAILURE);
    }

    /* Initialize host matrices with some data */
    for (int i = 0; i < size; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(size - i);
    }

    /* Allocate device memory */
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Configure kernel launch parameters */
    dim3 blockDim(16, 16);  /* 256 threads per block */
    dim3 gridDim((width  + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    /* Launch the kernel */
    matrixAddKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, width, height);
    CUDA_CHECK(cudaGetLastError());           /* Check for launch errors */
    CUDA_CHECK(cudaDeviceSynchronize());      /* Wait for kernel to finish */

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    int errors = 0;
    for (int y = 0; y < 5 && errors < 5; ++y) {
        for (int x = 0; x < 5 && errors < 5; ++x) {
            int idx = y * width + x;
            float expected = h_A[idx] + h_B[idx];
            if (fabs(h_C[idx] - expected) > 1e-5f) {
                fprintf(stderr,
                        "Mismatch at (%d,%d): GPU %f != CPU %f\n",
                        x, y, h_C[idx], expected);
                ++errors;
            }
        }
    }
    if (errors == 0)
        printf("Matrix addition verified successfully for sample entries.\n");
    else
        printf("Matrix addition found %d errors.\n", errors);

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
```