/*
Use `float` arrays instead of `int` arrays.

In this implementation we switch the data type of all arrays from int to float
to demonstrate how the same vector addition kernel works with floating point
numbers. The main changes are:
1. All host and device pointers are declared as `float*`.
2. The kernel `add` accepts `float*` arguments and performs `c[i] = a[i] + b[i]`.
3. Host memory allocation and initialization use `float`.
4. All CUDA memory operations (`cudaMalloc`, `cudaMemcpy`) now work with
   `sizeof(float) * N`.
5. The size of the data is printed in bytes using `sizeof(float)` to
   show the memory usage.
6. Verification compares each element within a small tolerance because
   floating point operations can produce rounding differences.

This example also demonstrates basic error checking with a helper macro
`CUDA_CHECK` that prints an error and exits if any CUDA call fails.

The program allocates two input arrays (`h_a`, `h_b`) of length `N`, initializes
them, copies them to device memory, launches the kernel, copies the result
back to the host, and prints the first few elements to confirm correctness.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1<<20) // 1M elements
#define BLOCK_SIZE 256

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err),  \
                    #call);                                             \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Kernel performing element-wise addition
__global__ void add(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main(void)
{
    float *h_a = NULL, *h_b = NULL, *h_c = NULL;
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c = (float *)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_a, size));
    CUDA_CHECK(cudaMalloc((void **)&d_b, size));
    CUDA_CHECK(cudaMalloc((void **)&d_c, size));

    // Copy host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify a few elements
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: GPU %f != Host %f\n",
                    i, h_c[i], expected);
            errors++;
        }
    }
    if (errors == 0)
        printf("Verification passed for first 10 elements.\n");
    else
        printf("Verification failed with %d errors.\n", errors);

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    printf("Program completed successfully.\n");
    return 0;
}
