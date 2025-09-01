/*
Matrices are stored in 1D memory (row-major). The 1D index is `idx = global_y * width + global_x`. Implement the matrix add using this.

Thinking:
- The goal is to write a complete CUDA program that adds two matrices element‑wise.
- The matrices are represented as 1‑D arrays in row‑major order. For an element at coordinates (global_y, global_x) the linear index is computed as `idx = global_y * width + global_x`.
- The kernel will calculate the global coordinates from block and thread indices, compute the linear index, and then perform the addition.
- The host will allocate host and device memory, initialize two input matrices, copy them to the device, launch the kernel, copy the result back, and finally verify the result by checking a few entries.
- Simple error checking macros are included to keep the code robust.
- The program uses single‑precision floats for the matrices. Block size is chosen as 16x16 which is a common tile size for such operations.
- After the kernel execution the program prints the first 5 elements of the result to confirm correct execution.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors following a kernel launch or API call */
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

/* Kernel performing matrix addition.
 * Each thread computes one element of the output matrix C = A + B.
 * The matrices are stored in 1D arrays in row-major order.
 */
__global__ void matAddKernel(const float *A, const float *B, float *C,
                             int width, int height)
{
    // Compute global 2D coordinates
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (global_x >= width || global_y >= height)
        return;

    // Compute linear index in row-major order
    int idx = global_y * width + global_x;

    // Perform addition
    C[idx] = A[idx] + B[idx];
}

int main()
{
    // Define matrix dimensions
    const int width  = 512;   // Number of columns
    const int height = 512;   // Number of rows
    const int numElements = width * height;
    const size_t sizeInBytes = numElements * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(sizeInBytes);
    float *h_B = (float *)malloc(sizeInBytes);
    float *h_C = (float *)malloc(sizeInBytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host matrices with some values
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = (float)i;            // Example: A[i] = i
        h_B[i] = (float)(2 * i);      // Example: B[i] = 2*i
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, sizeInBytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, sizeInBytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, sizeInBytes));

    // Copy input matrices from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeInBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeInBytes, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(16, 16); // 256 threads per block
    dim3 gridDim((width  + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    matAddKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, width, height);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeInBytes, cudaMemcpyDeviceToHost));

    // Verify a few elements
    int correct = 1;
    for (int i = 0; i < 5; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabsf(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: host %f, device %f, expected %f\n",
                    i, h_C[i], expected, expected);
            correct = 0;
        }
    }
    if (correct) {
        printf("Matrix addition successful. First 5 results:\n");
        for (int i = 0; i < 5; ++i) {
            printf("C[%d] = %f\n", i, h_C[i]);
        }
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
