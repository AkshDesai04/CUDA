```c
/*
Generalize the vector addition program to handle vectors of a large size, e.g., 1,048,576 elements. This will require launching multiple blocks. Each thread must calculate its unique global index using `blockIdx.x * blockDim.x + threadIdx.x` to work on the correct element.

Thinking:
1. **Size and Grid Configuration** – For a large vector (1,048,576 elements) we need multiple thread blocks. A common block size is 256 threads, which is a good fit for most GPUs. The grid size is computed as `(N + blockSize - 1) / blockSize` to cover all elements. The kernel itself will guard against out-of-bounds accesses by checking `idx < N`.
2. **Memory Allocation** – Allocate host memory for input vectors A and B, and the output vector C. Allocate corresponding device memory and copy inputs to device. After kernel execution, copy results back to host.
3. **Kernel Implementation** – The kernel receives pointers to device arrays and the size N. Each thread computes its global index `idx` using the formula given in the prompt and performs addition only if `idx < N`.
4. **Error Checking** – Wrap CUDA API calls with a macro that checks for errors and aborts with a message if anything goes wrong.
5. **Performance** – The program is straightforward and memory‑bound. For real workloads one might consider using pinned host memory, streams, or coalesced memory accesses, but the current focus is on correctness and scalability.
6. **Output** – For verification, print the first few elements of the result.

This design follows the CUDA programming model: allocate, copy, launch, synchronize, copy back, and free.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* Kernel for vector addition */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main(void)
{
    const int N = 1 << 20;              // 1,048,576 elements
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;   // arbitrary data
        h_B[i] = 2.0f;
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size));

    /* Copy host data to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Determine grid and block dimensions */
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    /* Launch kernel */
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());          // Check for launch errors
    CHECK_CUDA(cudaDeviceSynchronize());     // Wait for kernel completion

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Verify results (print first 10 elements) */
    printf("Result (first 10 elements):\n");
    for (int i = 0; i < 10; ++i)
        printf("C[%d] = %f\n", i, h_C[i]);

    /* Free device memory */
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    /* Free host memory */
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
```