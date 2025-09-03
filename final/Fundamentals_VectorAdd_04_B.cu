```cuda
/* 
Implement for `double` data type.
This program demonstrates how to perform a simple vector addition using CUDA with
double-precision floating point numbers. The goal is to provide a clear example
of allocating device memory, copying data between host and device, launching a
kernel, and retrieving results back to the host.

The steps are:
1. Parse a vector size N from the command line or use a default.
2. Allocate host arrays A, B, and C of type double.
3. Initialize A and B with some known values.
4. Allocate corresponding device arrays dA, dB, and dC.
5. Copy A and B to the device.
6. Launch a kernel that adds the two vectors element-wise.
7. Copy the result vector dC back to the host array C.
8. Verify the correctness of the addition.
9. Clean up all allocated memory.

The kernel uses a simple 1D grid of 1D blocks. Error checking is performed
after each CUDA API call to ensure correctness. The program also prints the
first few results for inspection.

This example is intentionally straightforward so that the double data type
usage is clear, and it can be compiled with nvcc:
    nvcc -o vector_add vector_add.cu
and run as:
    ./vector_add 1000000
to add vectors of size 1,000,000.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Kernel to perform element-wise addition of two double vectors
__global__ void vectorAdd(const double *A, const double *B, double *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[])
{
    int N = 1 << 20; // Default to 1,048,576 elements
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Invalid vector size '%s'. Using default %d.\n", argv[1], N);
            N = 1 << 20;
        }
    }
    printf("Vector size: %d\n", N);

    size_t size = N * sizeof(double);

    // Allocate host memory
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = (double)i;
        h_B[i] = 2.0 * (double)i;
    }

    // Allocate device memory
    double *d_A = NULL;
    double *d_B = NULL;
    double *d_C = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    // Copy host memory to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        double expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-9) {
            fprintf(stderr, "Mismatch at index %d: GPU %f, CPU %f\n", i, h_C[i], expected);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Vector addition successful. First 10 results:\n");
        for (int i = 0; i < 10 && i < N; ++i) {
            printf("C[%d] = %f\n", i, h_C[i]);
        }
    } else {
        printf("Vector addition failed.\n");
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```