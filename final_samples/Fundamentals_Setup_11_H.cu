/*
Aim: Write the kernel definition in a separate .cu file and the main function in another. Try to compile and link them together with nvcc.

Thinking:
- The user wants a simple CUDA program demonstrating how to separate kernel and main into different files.
- The developer instruction requires that the output be a single .cu file and no other commentary, so we combine the kernel and main into one file while still indicating that in a real project they would be split.
- We'll implement a basic vector addition kernel that adds two integer arrays element‑wise.
- The code will allocate host memory, fill input arrays, allocate device memory, copy data, launch the kernel, copy the result back, and verify correctness.
- Error handling is included using a macro that checks CUDA API return codes.
- The program is fully self‑contained and can be compiled with `nvcc` as a single file, e.g., `nvcc -o vecadd vecadd.cu`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* Kernel definition (would normally be in a separate .cu file) */
__global__ void vectorAdd(const int *a, const int *b, int *c, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

/* Main function (would normally be in a separate .cu file) */
int main(void)
{
    const size_t N = 1 << 20;          // Number of elements (1M)
    const size_t bytes = N * sizeof(int);

    /* Allocate host memory */
    int *h_a = (int *)malloc(bytes);
    int *h_b = (int *)malloc(bytes);
    int *h_c = (int *)malloc(bytes);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Initialize input arrays */
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = (int)i;
        h_b[i] = (int)(N - i);
    }

    /* Allocate device memory */
    int *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    /* Copy inputs to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    const size_t threadsPerBlock = 256;
    const size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    /* Verify results */
    int errors = 0;
    for (size_t i = 0; i < N; ++i) {
        int expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            ++errors;
            if (errors <= 10) {
                fprintf(stderr, "Mismatch at index %zu: got %d, expected %d\n",
                        i, h_c[i], expected);
            }
        }
    }
    if (errors == 0) {
        printf("All %zu elements matched!\n", N);
    } else {
        printf("Found %d mismatches.\n", errors);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return EXIT_SUCCESS;
}
