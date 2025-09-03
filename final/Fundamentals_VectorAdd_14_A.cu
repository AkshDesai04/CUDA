```cuda
/*
 * Aim: Ensure all host and device allocations use sizeof(double).
 * 
 * Thinking:
 * The goal is to write a CUDA C program that demonstrates proper memory allocation
 * for double-precision data both on the host (CPU) and on the device (GPU).
 * All allocations should be performed using `sizeof(double)` multiplied by the
 * number of elements. This includes:
 *   - Host allocations using malloc or calloc.
 *   - Device allocations using cudaMalloc.
 * The program will perform a simple vector addition on double arrays to
 * showcase the allocation and data transfer process. 
 * We include error checking for CUDA calls and memory allocation failures,
 * and we free all allocated memory before exiting.
 * 
 * The structure of the program:
 *   1. Define a simple kernel `add` that adds two double arrays elementwise.
 *   2. In `main`:
 *        - Set the vector size `N`.
 *        - Allocate host arrays `h_a`, `h_b`, `h_c` with `malloc(sizeof(double)*N)`.
 *        - Initialize `h_a` and `h_b` with some values.
 *        - Allocate device arrays `d_a`, `d_b`, `d_c` with `cudaMalloc`.
 *        - Copy `h_a` and `h_b` to device memory.
 *        - Launch kernel with appropriate grid and block dimensions.
 *        - Copy result back to `h_c`.
 *        - Verify correctness by printing a few elements.
 *   3. Free host and device memory.
 * 
 * All sizes are calculated using `sizeof(double)` to ensure consistency
 * and correctness when changing data type or precision.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in file '%s' in line %d: %s.\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/* Kernel for element-wise addition of two double arrays */
__global__ void add(const double *a, const double *b, double *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    /* Vector size */
    const int N = 1 << 20; /* 1M elements */
    const size_t bytes = N * sizeof(double);

    /* Host allocations using sizeof(double) */
    double *h_a = (double *)malloc(bytes);
    double *h_b = (double *)malloc(bytes);
    double *h_c = (double *)malloc(bytes);

    if (h_a == NULL || h_b == NULL || h_c == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<double>(i);
        h_b[i] = static_cast<double>(2 * i);
    }

    /* Device allocations using sizeof(double) */
    double *d_a = NULL;
    double *d_b = NULL;
    double *d_c = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /* Determine grid and block sizes */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    /* Verify results by printing first 5 elements */
    printf("Result of addition (first 5 elements):\n");
    for (int i = 0; i < 5; ++i) {
        printf("h_c[%d] = %f\n", i, h_c[i]);
    }

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    /* Free host memory */
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```