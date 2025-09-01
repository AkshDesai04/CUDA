/*
Aim: Use `float` arrays instead of `int` arrays.

Thinking:
The task is to produce a simple CUDA program that demonstrates working with floating-point arrays rather than integer arrays. A typical example is vector addition. I will implement a kernel that adds two float vectors element-wise and stores the result in a third vector. The host code will:
1. Allocate and initialize two float arrays on the host.
2. Allocate memory on the device and transfer the host data.
3. Launch the kernel with an appropriate grid/block configuration.
4. Transfer the result back to the host.
5. Verify the result and free all allocated memory.

I will also include a small error‑checking helper to make debugging easier. The program will be fully self‑contained, compile with `nvcc`, and produce a simple output verifying the computation. This fulfills the requirement to use `float` arrays and to produce a .cu file. The entire code is wrapped in a single file.*/

#include <stdio.h>
#include <cuda.h>

#define CHECK_CUDA(call)                                                    \
    {                                                                        \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    }

// Kernel to add two float vectors
__global__ void vecAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20;  // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f * i;
        h_B[i] = 2.0f * i;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    // Copy host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());  // Check kernel launch

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify result for a few elements
    bool success = true;
    for (int i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            fprintf(stderr, "Mismatch at index %d: got %f, expected %f\n",
                    i, h_C[i], expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector addition successful for first 10 elements.\n");
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
