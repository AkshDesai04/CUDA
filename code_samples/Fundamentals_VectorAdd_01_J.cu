/*
Add error checking to all CUDA calls and the kernel launch.

The goal of this program is to demonstrate how to perform basic vector addition using CUDA, with comprehensive error checking on every CUDA API call and the kernel launch itself.  By wrapping each CUDA function call in a macro that verifies the returned cudaError_t, we can immediately detect and report problems such as memory allocation failures, kernel launch failures, or data transfer errors.  The kernel itself performs a simple element‑wise addition of two input arrays.  After the kernel finishes, cudaGetLastError() and cudaDeviceSynchronize() are used to capture any errors that occurred during execution.  The program also checks for errors after copying data back to the host and after freeing device memory.  This pattern is widely used in CUDA applications to ensure robust error handling and easier debugging.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro to check CUDA API return values */
#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);           \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

/* Kernel for vector addition: C = A + B */
__global__ void vecAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20;          // Number of elements (1M)
    size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    /* Copy data from host to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Check for launch errors */
    CHECK_CUDA(cudaGetLastError());

    /* Wait for GPU to finish */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Verify result */
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            fprintf(stderr, "Mismatch at index %d: expected %f, got %f\n",
                    i, expected, h_C[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Vector addition succeeded.\n");
    }

    /* Free device memory */
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    /* Free host memory */
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
