/*
The original SAXPY is in-place, modifying Y. Implement this version: Y[i] = a * X[i] + Y[i].

Thinking:
- We need a CUDA program in C that performs the SAXPY operation in-place on array Y.
- We'll write a kernel that takes pointers to X and Y, the scalar a, and the number of elements N.
- Each thread will handle one element: Y[i] = a * X[i] + Y[i].
- On the host, we allocate host arrays Xh and Yh, initialize them, copy to device.
- Launch the kernel with appropriate grid/block dimensions.
- Copy the result back to host and print a few elements to verify.
- Include error checking for CUDA calls.
- Provide a simple main that demonstrates the functionality for a fixed array size.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// SAXPY kernel: Y[i] = a * X[i] + Y[i]
__global__ void saxpy_kernel(const float *X, float *Y, float a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Y[idx] = a * X[idx] + Y[idx];
    }
}

int main(void) {
    const int N = 1 << 20;          // 1M elements
    const float a = 2.5f;           // scaling factor

    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *Xh = (float *)malloc(bytes);
    float *Yh = (float *)malloc(bytes);
    if (!Xh || !Yh) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        Xh[i] = 1.0f * i;  // X[i] = i
        Yh[i] = 2.0f * i;  // Y[i] = 2i
    }

    // Allocate device memory
    float *Xd, *Yd;
    CUDA_CHECK(cudaMalloc((void **)&Xd, bytes));
    CUDA_CHECK(cudaMalloc((void **)&Yd, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(Xd, Xh, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Yd, Yh, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(Xd, Yd, a, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(Yh, Yd, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Sample results:\n");
    for (int i = 0; i < 10; ++i) {
        float expected = a * Xh[i] + (2.0f * i);
        printf("Y[%d] = %f  (expected %f)\n", i, Yh[i], expected);
    }

    // Clean up
    CUDA_CHECK(cudaFree(Xd));
    CUDA_CHECK(cudaFree(Yd));
    free(Xh);
    free(Yh);

    return EXIT_SUCCESS;
}
