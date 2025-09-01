```cuda
/* 
Change the data type from `int` to `float` and initialize the host array with floating-point values.
This comment states the exact aim of the program as requested. In addition, I explain my reasoning here:

1. I keep the overall structure of a basic CUDA program that performs vector addition on the GPU.
2. The original version presumably used `int` types; I replace all `int` occurrences with `float`.
3. The host arrays `h_A` and `h_B` are initialized with floating‑point literals (e.g., 1.0f, 2.0f, …).
4. The CUDA kernel `vectorAdd` now operates on `float` pointers and performs floating‑point addition.
5. Memory allocation, copying, kernel launch, and result retrieval all use `float` sizes.
6. A simple error‑checking macro `CHECK_CUDA` is added for robustness.
7. The program prints the resulting vector to demonstrate successful execution.

With these changes, the program compiles with `nvcc` and correctly demonstrates the requested data type change and initialization. 
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                  \
    {                                                                     \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

/* CUDA kernel for element‑wise vector addition of floats */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    /* Problem size */
    int N = 10;                       // Number of elements
    size_t size = N * sizeof(float);  // Size in bytes

    /* Host allocations */
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    /* Initialize host arrays with floating‑point values */
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i + 0.5f;            // e.g., 0.5, 1.5, 2.5, ...
        h_B[i] = (float)(N - i) + 1.0f;      // e.g., 11.0, 10.0, ...
    }

    /* Device allocations */
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    /* Copy inputs from host to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Launch kernel: one block of 256 threads */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError()); // Check for launch errors

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Print the result */
    printf("Result of vector addition (float values):\n");
    for (int i = 0; i < N; ++i)
    {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    /* Clean up */
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```