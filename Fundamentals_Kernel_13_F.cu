```cuda
/*
 * Implement element-wise multiplication: `c[i] = a[i] * b[i]`.
 *
 * Thinking:
 * The goal is to provide a complete, self‑contained CUDA program that demonstrates
 * element‑wise multiplication of two arrays.  We need to allocate input arrays `a`
 * and `b`, perform the multiplication on the GPU, and retrieve the result array
 * `c`.  The program will:
 *   1. Define a kernel that takes three pointers and an array size `N`.
 *   2. In `main`, allocate host memory, initialize `a` and `b` with sample data.
 *   3. Allocate device memory and copy the host arrays to device.
 *   4. Launch the kernel with a reasonable grid/block configuration.
 *   5. Copy the result back to the host, verify a few entries, and free memory.
 *
 * To keep the example portable, we use `float` arrays and a fixed size `N = 1 << 20`
 * (1,048,576 elements).  The kernel uses a simple 1D indexing scheme.  We include
 * basic error checking with a helper macro `CUDA_CHECK`.  The code is fully
 * compilable with `nvcc` and will print a few sample results to verify correctness.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",        \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),            \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel: element-wise multiplication
__global__ void elementwise_mul(const float *a, const float *b, float *c, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] * b[idx];
    }
}

int main(void)
{
    const size_t N = 1 << 20;          // 1,048,576 elements
    const size_t SIZE = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(SIZE);
    float *h_b = (float *)malloc(SIZE);
    float *h_c = (float *)malloc(SIZE);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = (float)i;          // Example values: 0, 1, 2, ...
        h_b[i] = (float)(N - i);    // Example values: N, N-1, N-2, ...
    }

    // Allocate device memory
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_a, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_b, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_c, SIZE));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, SIZE, cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    const int THREADS_PER_BLOCK = 256;
    int blocks = (int)((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // Launch kernel
    elementwise_mul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, SIZE, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Sample results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("c[%d] = %f (expected %f)\n", i, h_c[i], h_a[i] * h_b[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```