```cuda
/*
Implement vector addition using a 'grid-stride loop' inside the kernel. Instead of each thread processing only one element, launch a smaller grid of threads and have each thread process multiple elements in a strided loop: `for (int i = thread_id; i < N; i += grid_size) { ... }`. This makes the kernel independent of the number of threads launched.

My thinking while writing this program:
1. I need a kernel that can add two float vectors A and B into C, but I want to use a grid‑stride loop so the kernel does not depend on the number of threads. 
2. The thread id will be computed as `int tid = blockIdx.x * blockDim.x + threadIdx.x;`. The total grid size is `int gridSize = blockDim.x * gridDim.x;`.
3. Inside the kernel I will loop over indices: `for (int i = tid; i < N; i += gridSize) { C[i] = A[i] + B[i]; }`.
4. On the host, I will allocate three arrays of size N, fill A and B with some data, copy them to device, run the kernel, copy C back, and verify the result.
5. I will use a simple error‑checking macro `CUDA_CHECK` that prints any CUDA error and exits.
6. I will use a block size of 256 threads and compute the number of blocks as `(N + blockSize - 1) / blockSize`, but since the kernel is independent of the number of threads, it will still work if I under‑provision the grid (e.g., fewer blocks). For clarity I will keep the grid sized to cover the array but the code will work even if the grid is smaller.
7. I will include `<stdio.h>` and `<stdlib.h>` and compile with `nvcc`.
8. I will make the code self‑contained, with no external dependencies, and add comments for clarity.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

/* Macro to check CUDA API calls */
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

/* Kernel performing vector addition using a grid‑stride loop */
__global__ void vecAddGridStride(const float *A, const float *B, float *C, int N)
{
    /* Compute the global thread index and the total number of threads in the grid */
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    /* Each thread processes multiple elements, stepping by the grid stride */
    for (int i = tid; i < N; i += stride)
    {
        C[i] = A[i] + B[i];
    }
}

int main(void)
{
    /* Size of the vectors (number of elements) */
    const int N = 1 << 20;          /* 1,048,576 elements */
    const size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    /* Copy host vectors to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Kernel launch configuration */
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch the kernel */
    vecAddGridStride<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Check for launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verify a few elements */
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: host %f, device %f\n",
                    i, expected, h_C[i]);
            errors++;
        }
    }
    if (errors == 0)
        printf("Vector addition succeeded. Sample output:\n");
    else
        printf("Vector addition failed with %d mismatches.\n", errors);

    for (int i = 0; i < 10; ++i)
        printf("C[%d] = %f\n", i, h_C[i]);

    /* Clean up */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```