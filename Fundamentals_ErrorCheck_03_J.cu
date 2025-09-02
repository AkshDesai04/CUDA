/*
Refactor your vector add program to include post-launch error checking.

My thinking process for this refactoring was as follows:

1. The original vector addition program typically performs CUDA memory allocation, memory copy, kernel launch, and a final memory copy back to host, but often it only checks for errors immediately after each API call. The problem is that errors may not surface until after the kernel launch, especially if the kernel itself has issues. Therefore, we need to add error checking immediately after the kernel launch as well.

2. In CUDA, error checking after a kernel launch requires calling `cudaGetLastError()` to capture any launch errors. Additionally, to ensure that the kernel has finished executing and to catch errors that occur during execution, we should call `cudaDeviceSynchronize()` and check its return status.

3. To keep the code clean, I decided to create a helper macro `CUDA_CHECK` that takes an expression, executes it, checks the returned `cudaError_t`, and if an error occurs, prints the error message and aborts the program.

4. I included error checks after each `cudaMalloc`, `cudaMemcpy`, and after the kernel launch (`cudaGetLastError` and `cudaDeviceSynchronize`). This guarantees that any failure point is reported immediately with a meaningful message.

5. The rest of the program remains unchanged: a simple element‑wise addition kernel, host memory allocation, data initialization, device memory copy, kernel execution, result copy back, verification, and cleanup.

6. The output file is a single .cu file with the entire code, starting with a multiline comment that contains the aim and the thinking.

7. No extraneous text or files are produced; only the .cu file is output.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Kernel for vector addition: C = A + B */
__global__ void vecAdd(const float *A, const float *B, float *C, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

int main(void)
{
    /* Size of vectors */
    const int N = 1 << 20; /* 1M elements */
    const int size = N * sizeof(float);

    /* Host allocations */
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        exit(EXIT_FAILURE);
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    /* Device allocations */
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    /* Copy input data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Post-launch error checking */
    cudaError_t err = cudaGetLastError(); /* Check for launch errors */
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaDeviceSynchronize(); /* Wait for kernel to finish and check for execution errors */
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Verify result */
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            fprintf(stderr, "Verification failed at index %d: %f != %f\n", i, h_C[i], expected);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Vector addition successful, all results are correct.\n");
    }

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
