/*
Handle large vectors with boundary checks.

I chose to implement a simple element-wise addition of two large vectors
into a third vector. This demonstrates how to process data that is larger
than a single kernel launch might comfortably handle, by letting the
CUDA grid cover the entire range of indices. A boundary check inside
the kernel guarantees that no thread accesses out-of-range memory.
The program includes:
  - Host-side allocation and initialization of two large input vectors.
  - Device-side allocation using cudaMalloc and copying data with
    cudaMemcpy.
  - A CUDA kernel that checks if the global thread index is less than
    the vector size before performing the addition.
  - CUDA error checking macros to catch and report errors.
  - Timing using cudaEvent to show performance on the GPU.
  - A simple validation of the first few results on the host.
All the code is contained in a single .cu file and can be compiled with
`nvcc -o vector_add vector_add.cu` and run as `./vector_add [size]`.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* CUDA error checking macro */
#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",    \
                    #call, __FILE__, __LINE__,                   \
                    cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

/* Kernel performing element-wise addition with boundary check */
__global__ void vectorAddKernel(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char *argv[])
{
    /* Default vector size: 1 << 28 (~268 million elements) */
    size_t N = (1 << 28);
    if (argc > 1) {
        N = strtoull(argv[1], NULL, 10);
        if (N == 0) {
            fprintf(stderr, "Invalid vector size: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
    }

    printf("Vector size: %zu (%.2f MB per vector)\n", N, (double)N * sizeof(float) / (1024 * 1024));

    /* Allocate host memory */
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_C = (float *)malloc(N * sizeof(float));
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    /* Allocate device memory */
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_C, N * sizeof(float)));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (int)((N + threadsPerBlock - 1) / threadsPerBlock);
    printf("Launching kernel with %d blocks of %d threads.\n", blocksPerGrid, threadsPerBlock);

    /* Timing events */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, (int)N);
    CUDA_CHECK(cudaGetLastError()); // Check kernel launch error

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Compute elapsed time */
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel execution time: %.3f ms\n", ms);

    /* Simple verification */
    int errors = 0;
    for (size_t i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            printf("Mismatch at index %zu: got %f, expected %f\n", i, h_C[i], expected);
            errors++;
        }
    }
    if (errors == 0) {
        printf("First 10 elements verified successfully.\n");
    }

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
