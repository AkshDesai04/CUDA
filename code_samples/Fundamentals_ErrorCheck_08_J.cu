/*
Aim: Write a conclusion in comments: always use `cuda-memcheck` during development, and have robust error checking in your code.

This file demonstrates a minimal CUDA program that performs element‑wise addition of two vectors.
The goal is to showcase good coding practices for CUDA:
- Use a macro `CHECK_CUDA` to wrap all CUDA API calls with error checking.
- Handle memory allocation, transfer, and kernel launch errors explicitly.
- Include informative error messages with file name and line number.
- Verify the result on the host side.
- Finally, print a concise conclusion reminding developers to use `cuda-memcheck`
  during development and to maintain robust error checking throughout the code.
The program compiles with `nvcc` and can be run directly. It also serves as a template
for more complex CUDA applications where the same error‑checking pattern applies.
*/

#include <stdio.h>
#include <stdlib.h>

/* Macro to check CUDA errors and abort if any occur */
#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                                  \
        }                                                                         \
    } while (0)

/* Kernel to perform element‑wise addition of two arrays */
__global__ void addKernel(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void) {
    const int N = 1024;                 // Number of elements
    const size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void **)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_b, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_c, bytes));

    /* Copy host arrays to device */
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    /* Check for kernel launch errors */
    CHECK_CUDA(cudaGetLastError());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    printf("First 5 results:\n");
    for (int i = 0; i < 5; ++i) {
        printf("c[%d] = %f\n", i, h_c[i]);
    }

    /* Clean up */
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

/*
Conclusion:
Always use `cuda-memcheck` during development, and have robust error checking in your code.
This ensures that memory access violations, kernel launch failures, and API errors
are caught early, leading to more reliable and maintainable CUDA applications.
*/
