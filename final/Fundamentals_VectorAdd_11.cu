/*
Implement vector addition using a 'grid-stride loop' inside the kernel.
Instead of each thread processing only one element, launch a smaller grid of threads and
have each thread process multiple elements in a strided loop:
`for (int i = thread_id; i < N; i += grid_size) { ... }`.
This makes the kernel independent of the number of threads launched.

Thinking:
- A classic vector addition kernel assigns one element per thread.
  That requires launching at least N threads; if N is huge we might run out
  of thread slots or waste resources.
- Using a grid‑stride loop lets us launch a modest number of threads
  (e.g., 256 threads per block) and have each thread loop over the
  remaining elements by jumping ahead by the total number of threads
  in the grid (`stride = blockDim.x * gridDim.x`).
- The kernel obtains the global thread index (`idx`) and then
  iterates: `for (int i = idx; i < N; i += stride)`.
  This pattern is safe because each thread visits distinct indices.
- The kernel is independent of the launch configuration: changing
  the grid or block size automatically adjusts the stride, so the
  same code works for N = 10 or N = 10^9.
- The host code allocates device memory, copies inputs,
  launches the kernel with a chosen grid (e.g., 64 blocks of 256
  threads), copies back the result, and checks a few values.
- Error checking is performed with a helper macro `CUDA_CHECK`.
- The example uses float vectors and a size of 1<<20 (1M) elements,
  but it can be changed easily.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",                \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

/* Kernel performing vector addition with a grid‑stride loop */
__global__ void vecAdd(const float *a, const float *b, float *c, int N)
{
    /* Global thread index */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    /* Total number of threads in the grid */
    int stride = blockDim.x * gridDim.x;

    /* Each thread processes multiple elements by stepping through the array */
    for (int i = idx; i < N; i += stride)
    {
        c[i] = a[i] + b[i];
    }
}

int main(void)
{
    /* Size of vectors */
    const int N = 1 << 20;  /* 1,048,576 elements */
    const size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input vectors */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    /* Copy input data from host to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /* Define block and grid dimensions */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = 64;  /* Can be any value; grid‑stride loop adapts */

    /* Launch kernel */
    dim3 blockSize(threadsPerBlock);
    dim3 gridSize(blocksPerGrid);
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    /* Check for launch errors */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: host=%f, device=%f, expected=%f\n",
                    i, h_c[i], expected, h_c[i]);
            errors++;
        }
    }
    if (errors == 0) {
        printf("Vector addition successful. First 10 results:\n");
        for (int i = 0; i < 10; ++i) {
            printf("c[%d] = %f\n", i, h_c[i]);
        }
    } else {
        printf("Vector addition failed with %d errors.\n", errors);
    }

    /* Clean up */
    free(h_a);
    free(h_b);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
