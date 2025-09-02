/*
Aim of the program: Handle large vectors with boundary checks.

Thinking:
The goal is to demonstrate how to work with large vectors in CUDA while ensuring that each thread only accesses valid memory locations. 
To achieve this, the program will:
1. Allocate large vectors on the host.
2. Allocate corresponding memory on the device.
3. Copy data from host to device.
4. Launch a CUDA kernel that performs an element-wise operation (vector addition) with a boundary check (`if (idx < N)`).
5. Copy the result back to the host.
6. Verify the result on the host side.
7. Include error checking for CUDA API calls.
8. Use dynamic grid and block dimensions that can handle sizes that are not multiples of the block size.
9. Print a simple success message after verification.

The code is written in C/C++ style for CUDA (.cu) and can be compiled with `nvcc`. It demonstrates handling vectors that could be much larger than the grid dimension by using boundary checks to avoid out-of-bounds memory access. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Macro for CUDA error checking
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Kernel for element-wise vector addition with boundary check
__global__ void vectorAdd(const float *a, const float *b, float *c, size_t N)
{
    // Compute global index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Boundary check
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host function to initialize vector with a pattern
void initVector(float *v, size_t N, float value)
{
    for (size_t i = 0; i < N; ++i) {
        v[i] = value + i;  // simple pattern
    }
}

// Host function to verify result
int verifyResult(const float *a, const float *b, const float *c, size_t N)
{
    for (size_t i = 0; i < N; ++i) {
        float expected = a[i] + b[i];
        if (fabs(c[i] - expected) > 1e-5) {
            fprintf(stderr, "Verification failed at index %zu: %f + %f != %f\n",
                    i, a[i], b[i], c[i]);
            return 0;
        }
    }
    return 1;
}

int main()
{
    // Size of vectors (large vector)
    size_t N = 100 * 1000 * 1000;  // 100 million elements (~400 MB per vector)

    printf("Vector size: %zu elements (%.2f MB per vector)\n", N, (double)(N * sizeof(float)) / (1024.0 * 1024.0));

    // Allocate host memory
    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_c = (float *)malloc(N * sizeof(float));
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    initVector(h_a, N, 1.0f);
    initVector(h_b, N, 2.0f);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void **)&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_c, N * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    // Define block and grid sizes
    const int threadsPerBlock = 256;
    int blocksPerGrid = (int)((N + threadsPerBlock - 1) / threadsPerBlock);

    printf("Launching kernel with %d blocks of %d threads each.\n", blocksPerGrid, threadsPerBlock);

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result
    if (verifyResult(h_a, h_b, h_c, N)) {
        printf("Verification successful: all elements match expected results.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}
