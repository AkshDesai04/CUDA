/*
Aim: Handle `double` precision data.

Thinking:
The user requested a CUDA program that handles double precision data.  In CUDA, double precision arithmetic is supported on GPUs with compute capability 1.3 and higher.  The simplest illustration is a vector addition kernel that takes two double arrays, adds them element‑wise, and writes the result to a third array.  
To make the example self‑contained, the host code will:
1. Allocate host vectors of type double.
2. Initialize them with sample values.
3. Allocate device memory for the vectors.
4. Copy the host data to the device.
5. Launch a kernel that performs double‑precision addition.
6. Copy the result back to the host.
7. Verify the result and print a few elements.
The code includes basic error checking and uses the CUDA runtime API.  It is written in C/C++ and compiled as a `.cu` file.  No external libraries are required beyond CUDA.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a kernel launch or API call
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel that adds two double arrays element‑wise
__global__ void vectorAdd(const double *a, const double *b, double *c, size_t n)
{
    // Compute global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    const size_t N = 1 << 20;   // 1M elements
    const size_t bytes = N * sizeof(double);

    // Allocate host memory
    double *h_a = (double *)malloc(bytes);
    double *h_b = (double *)malloc(bytes);
    double *h_c = (double *)malloc(bytes);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = 1.0;          // arbitrary values
        h_b[i] = 2.0;
    }

    // Allocate device memory
    double *d_a = nullptr;
    double *d_b = nullptr;
    double *d_c = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (int)((N + threadsPerBlock - 1) / threadsPerBlock);

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());   // Check for kernel launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify results (print first 5 elements)
    printf("First 5 results of double precision vector addition:\n");
    for (int i = 0; i < 5; ++i) {
        printf("c[%d] = %.1f\n", i, h_c[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    printf("Completed double precision vector addition.\n");
    return EXIT_SUCCESS;
}
