```cuda
/*
Aim: Handle large vectors with boundary checks.

Thinking:
The goal of this CUDA program is to demonstrate how to safely process a very large vector on the GPU
by incorporating boundary checks in the kernel so that threads do not access memory outside the
allocated array.  We will:
1. Allocate a large host vector (on the order of 10^8 elements) and initialize it.
2. Allocate device memory and copy the data from host to device.
3. Define a simple element‑wise operation (scaling each element by a factor) that will be performed
   by a CUDA kernel.  The kernel will compute a global thread index and then guard the memory
   access with an `if (idx < N)` boundary check to avoid out‑of‑bounds reads/writes.
4. Launch the kernel with a grid that covers all elements.
5. Copy the processed data back to the host and optionally verify a few elements.
6. Measure execution time using CUDA events for performance insight.
7. Use proper error checking for all CUDA API calls.

This structure ensures that even when the vector size does not divide evenly into the chosen
block size, the kernel remains safe and correct. The boundary check is the key element that
prevents segmentation faults or corrupted data when the last block has fewer than the maximum
number of threads.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Utility macro for checking CUDA API calls */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* Kernel to scale a vector by a given factor with boundary check */
__global__ void scaleVector(float *d_vec, int N, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {          // Boundary check
        d_vec[idx] *= scale;
    }
}

int main()
{
    /* Size of the vector: 100 million elements (~400 MB) */
    const size_t N = 100000000; // 1e8
    const float scaleFactor = 2.0f;

    /* Allocate host memory */
    float *h_vec = (float *)malloc(N * sizeof(float));
    if (!h_vec) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vector */
    for (size_t i = 0; i < N; ++i) {
        h_vec[i] = (float)i;
    }

    /* Allocate device memory */
    float *d_vec = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_vec, N * sizeof(float)));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Define kernel launch parameters */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (int)((N + threadsPerBlock - 1) / threadsPerBlock);

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Record start event */
    CUDA_CHECK(cudaEventRecord(start, 0));

    /* Launch kernel */
    scaleVector<<<blocksPerGrid, threadsPerBlock>>>(d_vec, (int)N, scaleFactor);

    /* Record stop event */
    CUDA_CHECK(cudaEventRecord(stop, 0));

    /* Wait for kernel to finish */
    CUDA_CHECK(cudaEventSynchronize(stop));

    /* Compute elapsed time */
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    printf("Kernel execution time: %f ms\n", elapsedMs);

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_vec, d_vec, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Verify a few elements */
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        float expected = (float)i * scaleFactor;
        if (fabs(h_vec[i] - expected) > 1e-5f) {
            printf("Mismatch at index %d: host=%f, expected=%f\n", i, h_vec[i], expected);
            errors++;
        }
    }
    if (errors == 0) {
        printf("Verification passed for first 10 elements.\n");
    } else {
        printf("Verification failed with %d mismatches.\n", errors);
    }

    /* Cleanup */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_vec));
    free(h_vec);

    return EXIT_SUCCESS;
}
```