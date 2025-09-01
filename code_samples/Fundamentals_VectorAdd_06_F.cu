/*
Aim: Handle large vectors with boundary checks.

This program demonstrates how to perform element‑wise addition of two very large vectors on a CUDA-enabled GPU while ensuring that every thread checks whether its computed index is within the bounds of the data. The vector size chosen is large enough to require many thread blocks and to illustrate the use of dynamic grid dimensions. Boundary checks are essential because CUDA kernels launch more threads than elements in the vector when the vector size is not an exact multiple of the block size. Without these checks, out‑of‑bounds memory accesses could corrupt data or crash the program.

The program follows these steps:

1. **Define constants and helper macros** – including an error‑checking macro that wraps CUDA API calls.
2. **Allocate host memory** – three arrays (A, B, C) large enough to hold the vectors.
3. **Initialize input vectors** – simple arithmetic patterns are used to keep the code small and deterministic.
4. **Allocate device memory** – corresponding GPU buffers for A, B, and C.
5. **Copy inputs to device** – `cudaMemcpy` with appropriate directions.
6. **Launch kernel** – the kernel is launched with a grid size computed from the vector length and a fixed block size. Inside the kernel, each thread calculates its global index and performs a boundary check (`if (idx < N)`) before accessing the arrays.
7. **Synchronize and copy results back** – after kernel completion, results are copied back to host memory.
8. **Verify correctness** – a simple host‑side check confirms that the GPU computation matches the expected results.
9. **Clean up** – free all allocated memory on host and device.

Boundary checking is performed in the kernel by verifying that the global index is less than `N`. This guard protects against reading or writing beyond the allocated array bounds, which could otherwise lead to undefined behaviour or program crashes. The kernel can thus safely be launched with a grid that covers the entire vector length, even when the number of threads is not an exact multiple of the vector size.

The program is written in standard C/C++ with CUDA extensions, and is fully self‑contained in a single .cu file. It compiles with `nvcc` and can be run on any system with an appropriate CUDA toolkit and GPU.*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/* Kernel: element-wise vector addition with boundary check */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

/* Main function */
int main(void)
{
    /* Size of vectors: 2^28 ≈ 268 million elements */
    const int N = 1 << 28;          /* 268,435,456 elements */
    const size_t size = (size_t)N * sizeof(float);

    printf("Vector size: %d elements (%zu bytes each array)\n", N, size);

    /* Allocate host memory */
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    /* Initialize input vectors */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)(N - i) * 0.3f;
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    /* Copy input data to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Kernel launch configuration */
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching kernel with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    /* Launch kernel */
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Synchronize */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Verify a few elements */
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: GPU %f, CPU %f\n", i, h_C[i], expected);
            errors++;
        }
    }
    if (errors == 0)
        printf("Sample verification passed.\n");
    else
        printf("Sample verification failed with %d errors.\n", errors);

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    /* Free host memory */
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Program completed successfully.\n");
    return 0;
}
