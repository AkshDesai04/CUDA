/*
Implement `C[i] = A[i] * scalar`.

Thinking:
I need to write a CUDA program that multiplies each element of an input array A by a given scalar and stores the result in an output array C. The program will:
1. Define a CUDA kernel that takes pointers to A, C, the scalar, and the number of elements.
2. Launch the kernel with an appropriate grid/block configuration.
3. Allocate host memory, initialize input data, copy it to device, launch the kernel, copy the result back, and finally verify the output.

Error checking will be done for CUDA API calls. The program will include a simple test case in `main` to demonstrate functionality.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA error checking macro
#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel to compute C[i] = A[i] * scalar
__global__ void vectorScale(const float* __restrict__ A,
                            float* __restrict__ C,
                            float scalar,
                            size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * scalar;
    }
}

int main()
{
    const size_t N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);
    const float scalar = 2.5f;

    // Host allocation
    float *h_A = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input array
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.001f; // arbitrary values
    }

    // Device allocation
    float *d_A = nullptr;
    float *d_C = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorScale<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, scalar, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Simple verification
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        float expected = h_A[i] * scalar;
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %zu: host=%f, device=%f, expected=%f\n",
                    i, h_A[i], h_C[i], expected);
            ok = false;
            break;
        }
    }
    if (ok) {
        printf("Verification passed.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Cleanup
    free(h_A);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
