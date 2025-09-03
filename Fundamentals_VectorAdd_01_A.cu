```cpp
/* 
Use `float`s for the vectors instead of `int`s. 

This program demonstrates how to perform vector addition on the GPU using CUDA
with floating‑point data types.  The original assignment was to use integers,
but here we switch to `float`s to handle real numbers.  The steps are:

1. Allocate two input vectors (`h_a`, `h_b`) and one output vector (`h_c`) on the host.
2. Initialize the input vectors with sample floating‑point values.
3. Allocate corresponding memory on the device (`d_a`, `d_b`, `d_c`).
4. Copy the input vectors from host to device.
5. Launch a CUDA kernel (`vectorAdd`) that adds the elements of `a` and `b`
   and stores the result in `c`.
6. Copy the result vector back to the host.
7. Verify a few results and free all allocated memory.

The kernel uses a one‑dimensional grid of blocks and threads, where each
thread processes one element of the vectors.  Boundary checks are performed
so that the kernel can safely handle vector sizes that are not multiples of
the block size.

All memory allocations are checked for errors, and the program prints
a short message indicating success or failure.  The output shows the
first ten elements of the result vector for quick verification.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",     \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),          \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

/* Kernel for vector addition: c = a + b */
__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main(int argc, char *argv[])
{
    /* Vector size: default 1<<20 (~1 million elements) */
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 20);

    /* Allocate host memory */
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c = (float*)malloc(N * sizeof(float));
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input vectors */
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i) * 0.5f;
        h_b[i] = static_cast<float>(N - i) * 0.25f;
    }

    /* Allocate device memory */
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_c, N * sizeof(float)));

    /* Copy inputs to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Determine execution configuration */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());  // Check kernel launch

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Simple verification: print first 10 results */
    printf("First 10 results of c = a + b:\n");
    for (int i = 0; i < 10 && i < N; ++i)
        printf("c[%d] = %f\n", i, h_c[i]);

    /* Clean up */
    free(h_a);
    free(h_b);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    printf("Vector addition completed successfully.\n");
    return EXIT_SUCCESS;
}
```