/*
Aim of the program: Split a large vector addition task across two GPUs. Allocate half of the A and B vectors on GPU 0 and the other half on GPU 1. Launch a kernel on each GPU to process its half. The host must manage both devices, switching context with cudaSetDevice.

Thinking: The program uses the CUDA Runtime API to manage two GPUs. It allocates the full vectors in host memory, then for each GPU it:
1. Sets the device context with cudaSetDevice.
2. Allocates device memory for half of the vectors.
3. Copies the corresponding half of the host data to the device.
4. Launches a simple vector addition kernel that operates on that half.
5. Copies the result back to the appropriate half of the host result vector.
The program includes error checking, ensures at least two GPUs are available, and verifies correctness by printing a few sample outputs. It is self‑contained and ready to compile with nvcc.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20;          // Size of full vectors
    const int half = N / 2;         // Half size for each GPU

    // Allocate host memory
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_C = (float *)malloc(N * sizeof(float));
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Check for at least two GPUs
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        fprintf(stderr, "This program requires at least two GPUs. Found %d.\n", deviceCount);
        return EXIT_FAILURE;
    }

    // Process first half on GPU 0
    CUDA_CHECK(cudaSetDevice(0));

    float *d_A0, *d_B0, *d_C0;
    CUDA_CHECK(cudaMalloc((void **)&d_A0, half * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_B0, half * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_C0, half * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A0, h_A, half * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B0, h_B, half * sizeof(float), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int numBlocks0 = (half + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks0, blockSize>>>(d_A0, d_B0, d_C0, half);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_C, d_C0, half * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A0));
    CUDA_CHECK(cudaFree(d_B0));
    CUDA_CHECK(cudaFree(d_C0));

    // Process second half on GPU 1
    CUDA_CHECK(cudaSetDevice(1));

    float *d_A1, *d_B1, *d_C1;
    CUDA_CHECK(cudaMalloc((void **)&d_A1, half * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_B1, half * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_C1, half * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A1, h_A + half, half * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B1, h_B + half, half * sizeof(float), cudaMemcpyHostToDevice));

    const int numBlocks1 = (half + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks1, blockSize>>>(d_A1, d_B1, d_C1, half);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_C + half, d_C1, half * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A1));
    CUDA_CHECK(cudaFree(d_B1));
    CUDA_CHECK(cudaFree(d_C1));

    // Verify a few elements
    printf("Sample results:\n");
    for (int i = 0; i < 5; ++i) {
        printf("C[%d] = %f (expected %f)\n", i, h_C[i], h_A[i] + h_B[i]);
    }
    for (int i = N - 5; i < N; ++i) {
        printf("C[%d] = %f (expected %f)\n", i, h_C[i], h_A[i] + h_B[i]);
    }

    // Clean up host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
