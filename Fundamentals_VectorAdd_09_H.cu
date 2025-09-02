/*
 * Pass all scalars and vector pointers to the kernel.
 *
 * Thinking:
 * The goal of this example is to demonstrate how to pass both scalar
 * values (such as array size, scalar coefficients) and pointers to
 * device vectors into a CUDA kernel.  We will allocate two input
 * vectors (x and y) on the host, copy them to the device, and then
 * launch a kernel that computes out[i] = a * x[i] + b * y[i] for each
 * element.  Scalars a, b, and the array size N will be passed as
 * kernel arguments alongside the device pointers.  After the kernel
 * finishes, the result will be copied back to the host and printed.
 *
 * The code below is a minimal, self‑contained CUDA program that
 * showcases these concepts.  It includes basic error checking and
 * cleanup.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel that performs out[i] = a * x[i] + b * y[i]
__global__ void compute_kernel(const float *x, const float *y, float *out,
                               float a, float b, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = a * x[idx] + b * y[idx];
    }
}

int main(void)
{
    const int N = 10;                // Number of elements
    const float a = 2.0f;            // Scalar multiplier for x
    const float b = 3.0f;            // Scalar multiplier for y

    // Host memory allocation
    float *h_x = (float *)malloc(N * sizeof(float));
    float *h_y = (float *)malloc(N * sizeof(float));
    float *h_out = (float *)malloc(N * sizeof(float));

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_x[i] = (float)i;
        h_y[i] = (float)(N - i);
    }

    // Device memory allocation
    float *d_x = NULL, *d_y = NULL, *d_out = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_out, N * sizeof(float)));

    // Copy host vectors to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    compute_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_out, a, b, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    printf("Result of out[i] = a * x[i] + b * y[i] with a = %.2f, b = %.2f:\n", a, b);
    for (int i = 0; i < N; ++i) {
        printf("out[%d] = %.2f\n", i, h_out[i]);
    }

    // Clean up
    free(h_x);
    free(h_y);
    free(h_out);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
