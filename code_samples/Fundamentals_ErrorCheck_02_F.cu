/*
Aim of the program:
Place all CUDA calls in your program on a separate line so the CHECK macro can wrap them cleanly.

Thinking:
- Create a simple vector addition program to demonstrate placing every CUDA API call on its own line.
- Define a CHECK macro that verifies the return value of each CUDA call and prints an error if it fails.
- The kernel launch is written on a single line and wrapped with the CHECK macro, with a separate CHECK for cudaGetLastError().
- All memory allocations, copies, kernel launch, and deallocations are each on distinct lines.
- The program includes a kernel definition, host code, and proper error handling.
- This file is fully self‑contained and can be compiled with nvcc.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                                 \
    {                                                                               \
        cudaError_t err = (call);                                                   \
        if (err != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error: %s (err_num=%d) at %s:%d\n",               \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }

/* Kernel that adds two vectors */
__global__ void vecAdd(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(void)
{
    const int N = 1 << 20;          /* 1M elements */
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    CHECK( cudaMalloc((void **)&d_a, size) );
    CHECK( cudaMalloc((void **)&d_b, size) );
    CHECK( cudaMalloc((void **)&d_c, size) );

    /* Copy data from host to device */
    CHECK( cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice) );

    /* Launch kernel */
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    CHECK( vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N) );
    CHECK( cudaGetLastError() );      /* Check for launch errors */

    /* Copy result back to host */
    CHECK( cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost) );

    /* Verify result */
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            ++errors;
            if (errors <= 10) {
                fprintf(stderr, "Mismatch at index %d: got %f, expected %f\n",
                        i, h_c[i], expected);
            }
        }
    }
    if (errors == 0) {
        printf("All results correct.\n");
    } else {
        printf("%d mismatches found.\n", errors);
    }

    /* Free device memory */
    CHECK( cudaFree(d_a) );
    CHECK( cudaFree(d_b) );
    CHECK( cudaFree(d_c) );

    /* Free host memory */
    free(h_a);
    free(h_b);
    free(h_c);

    return EXIT_SUCCESS;
}
