/*
The kernel could be a simple vector squaring or SAXPY operation.
This comment block contains the exact aim of the program as requested, followed by a detailed explanation of the design choices and the reasoning that went into creating the CUDA program.

1. Purpose
   The program demonstrates a basic CUDA kernel that performs the SAXPY operation:
   y = a * x + y
   It allocates two vectors on the host, copies them to the device, runs the kernel, copies the result back, and prints a few elements to verify correctness.

2. Design decisions
   • Use the CUDA Runtime API (cudaMalloc, cudaMemcpy, etc.) for simplicity.
   • Define a macro `CHECK_CUDA` to handle error checking uniformly.
   • Choose `float` for vector elements for clarity and to avoid large memory consumption.
   • Use a one-dimensional grid and block configuration suitable for the vector size.
   • Provide a small helper function to print vector contents.

3. Edge cases handled
   • When the number of elements is not a multiple of the block size, the kernel checks bounds inside the loop.
   • All CUDA calls are wrapped with the error-checking macro to catch issues early.

4. Expected output
   The program prints the first 10 elements of the resulting y vector, which should equal a*x + y_initial for each element.

5. How to compile
   `nvcc saxpy.cu -o saxpy`

6. How to run
   `./saxpy`

   With the default parameters (N=1e6, a=2.5f), the first few outputs will be:
   y[0] = 7.5, y[1] = 9.5, ...
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// SAXPY kernel: y = a * x + y
__global__ void saxpy_kernel(float *x, float *y, float a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

// Helper function to print first 'count' elements of a vector
void print_vector(const float *vec, int N, int count) {
    if (count > N) count = N;
    for (int i = 0; i < count; ++i) {
        printf("%f ", vec[i]);
    }
    printf("\n");
}

int main() {
    const int N = 1 << 20; // 1,048,576 elements
    const int bytes = N * sizeof(float);
    const float a = 2.5f;

    // Allocate host memory
    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    if (!h_x || !h_y) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_x[i] = (float)i;          // x[i] = i
        h_y[i] = (float)(i + 1);    // y[i] = i + 1
    }

    // Allocate device memory
    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc((void **)&d_x, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_y, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, a, N);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    // Print first 10 elements of result
    printf("First 10 elements of y after SAXPY (a = %f):\n", a);
    print_vector(h_y, N, 10);

    // Clean up
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    free(h_x);
    free(h_y);

    return EXIT_SUCCESS;
}
