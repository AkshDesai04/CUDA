/*
Implement linear interpolation (lerp): C[i] = A[i] * (1.0 - t) + B[i] * t, where t is a scalar float.
I started by understanding that the goal is to provide a CUDA kernel that performs element‑wise linear interpolation between two float arrays A and B into an output array C using a scalar interpolation factor t. The kernel will be launched with a sufficient number of threads to cover all elements of the arrays, and each thread will compute the interpolated value for its own index if that index is within bounds.

To make the program self‑contained and testable, I decided to:
1. Define a fixed array size N (e.g., 10) and allocate host memory for A, B, and C.
2. Initialize A and B with sample data.
3. Allocate corresponding device memory and copy the host arrays to the device.
4. Launch the kernel with a typical block size (256) and compute the number of blocks needed.
5. Copy the result back to the host and print it for verification.
6. Clean up all allocated memory.

I added a simple macro for CUDA error checking to make debugging easier. The kernel checks bounds to avoid out‑of‑range writes. The code uses standard CUDA runtime API functions and is ready to compile with `nvcc`.

Note: The code is deliberately simple for clarity and educational purposes. In production code, you would add more robust error handling, dynamic sizing, and possibly template support for different data types.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                               \
    {                                                                  \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    }

__global__ void lerpKernel(const float *A, const float *B, float *C, float t, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * (1.0f - t) + B[idx] * t;
    }
}

int main(void) {
    const int N = 10;
    const float t = 0.3f;

    /* Host memory allocation */
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_C = (float *)malloc(N * sizeof(float));
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input arrays */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;         // Example: 0,1,2,...
        h_B[i] = (float)(N - i);   // Example: 10,9,8,...
    }

    /* Device memory allocation */
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_B, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_C, N * sizeof(float)));

    /* Copy data from host to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Launch kernel */
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    lerpKernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, t, N);
    CHECK_CUDA(cudaGetLastError()); // Check kernel launch errors
    CHECK_CUDA(cudaDeviceSynchronize()); // Ensure completion

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Print results */
    printf("A: ");
    for (int i = 0; i < N; ++i) printf("%0.2f ", h_A[i]);
    printf("\nB: ");
    for (int i = 0; i < N; ++i) printf("%0.2f ", h_B[i]);
    printf("\nC (t = %0.2f): ", t);
    for (int i = 0; i < N; ++i) printf("%0.2f ", h_C[i]);
    printf("\n");

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return EXIT_SUCCESS;
}
