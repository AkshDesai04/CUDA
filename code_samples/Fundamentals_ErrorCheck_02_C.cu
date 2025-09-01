/*
Aim of the program:
Wrap all `cudaFree` calls.

Thinking:
In CUDA programming, freeing device memory with `cudaFree` is an essential
operation to avoid memory leaks. However, the raw `cudaFree` call returns an
error code that is often ignored, which can lead to silent failures. By
wrapping `cudaFree` in a helper function, we can centralise error handling,
log failures, and make the code cleaner. This wrapper will:
1. Call `cudaFree` on the given pointer.
2. Check the returned `cudaError_t`.
3. If an error occurs, print a descriptive message and abort execution.
4. Return `cudaSuccess` if everything is fine.

The program below demonstrates the usage of this wrapper. It allocates
device memory, performs a simple vector addition kernel, and frees the
memory using the wrapped `cudaFree`. The wrapper ensures that any failure
during deallocation is caught early and reported.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Wrapper for cudaFree that checks for errors */
void safeCudaFree(void *ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* Simple vector addition kernel */
__global__ void vecAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    /* Host allocations */
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    /* Initialize host data */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Device allocations */
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaError_t err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_A failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_B failed: %s\n", cudaGetErrorString(err));
        safeCudaFree(d_A);
        return EXIT_FAILURE;
    }
    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_C failed: %s\n", cudaGetErrorString(err));
        safeCudaFree(d_A);
        safeCudaFree(d_B);
        return EXIT_FAILURE;
    }

    /* Copy data from host to device */
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy h_A->d_A failed: %s\n", cudaGetErrorString(err));
        safeCudaFree(d_A); safeCudaFree(d_B); safeCudaFree(d_C);
        return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy h_B->d_B failed: %s\n", cudaGetErrorString(err));
        safeCudaFree(d_A); safeCudaFree(d_B); safeCudaFree(d_C);
        return EXIT_FAILURE;
    }

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Check for kernel launch errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        safeCudaFree(d_A); safeCudaFree(d_B); safeCudaFree(d_C);
        return EXIT_FAILURE;
    }

    /* Copy result back to host */
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_C->h_C failed: %s\n", cudaGetErrorString(err));
        safeCudaFree(d_A); safeCudaFree(d_B); safeCudaFree(d_C);
        return EXIT_FAILURE;
    }

    /* Verify a few elements */
    for (int i = 0; i < 5; ++i) {
        printf("h_C[%d] = %f\n", i, h_C[i]);
    }

    /* Free device memory using wrapped cudaFree */
    safeCudaFree(d_A);
    safeCudaFree(d_B);
    safeCudaFree(d_C);

    /* Free host memory */
    free(h_A);
    free(h_B);
    free(h_C);

    /* Reset device (optional) */
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
