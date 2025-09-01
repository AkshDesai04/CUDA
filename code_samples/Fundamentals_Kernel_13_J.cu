```cuda
/*
 * Chain kernel calls. The output of the first kernel (b) becomes the input to a second kernel that, for example, squares every element.
 *
 * Thinking process:
 *
 * 1. Define a simple demonstration where we have an input array `a` on the host, copy it to the device, and run a first kernel
 *    that writes into array `b`.  For illustration, `b[i] = a[i] * 2 + i`.  This shows some computation that depends on both
 *    the input value and the index.
 *
 * 2. After the first kernel finishes (we synchronize the device), launch a second kernel that takes array `b` as input
 *    and writes into array `c` the square of each element (`c[i] = b[i] * b[i]`).  This demonstrates chaining: the output of
 *    kernel1 feeds into kernel2.
 *
 * 3. Allocate memory on both host and device, copy data where necessary, launch kernels with a typical 1D grid configuration,
 *    synchronize after each launch, then copy the final results back to the host for verification.
 *
 * 4. Include basic CUDA error checking to make the example robust and easy to debug.
 *
 * 5. The code is written in plain C (no C++ features) and is suitable for compilation with `nvcc`.  It uses standard
 *    CUDA runtime API calls and prints a few elements of the result to demonstrate correctness.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for CUDA error checking */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",     \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),          \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* Kernel 1: Compute b[i] = a[i] * 2 + i */
__global__ void kernel1(const float *a, float *b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        b[idx] = a[idx] * 2.0f + (float)idx;
    }
}

/* Kernel 2: Compute c[i] = b[i] * b[i] (square of b) */
__global__ void kernel2(const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = b[idx] * b[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;          /* 1 million elements */
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    if (!h_a || !h_c)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input array on host */
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = (float)i;
    }

    /* Allocate device memory */
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_a, size));
    CUDA_CHECK(cudaMalloc((void **)&d_b, size));
    CUDA_CHECK(cudaMalloc((void **)&d_c, size));

    /* Copy input data from host to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    /* Define block and grid sizes */
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    /* Launch Kernel 1 */
    kernel1<<<gridSize, blockSize>>>(d_a, d_b, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Launch Kernel 2 */
    kernel2<<<gridSize, blockSize>>>(d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    /* Verify and print a few results */
    printf("First 10 results of c[i] = (a[i]*2 + i)^2:\n");
    for (int i = 0; i < 10; ++i)
    {
        float expected = (h_a[i] * 2.0f + (float)i);
        expected = expected * expected;
        printf("c[%d] = %f  (expected %f)\n", i, h_c[i], expected);
    }

    /* Clean up */
    free(h_a);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
```