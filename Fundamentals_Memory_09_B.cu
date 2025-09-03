/*
 * Remember to use `cudaFreeHost` to free memory allocated with `cudaMallocHost`.
 *
 * Thinking:
 * The task requires a CUDA C/C++ program that demonstrates the correct
 * allocation of pinned (page‑locked) host memory using cudaMallocHost
 * and its deallocation with cudaFreeHost.  
 * To make the example concrete, the program performs a simple vector
 * addition kernel.  Host vectors A and B are allocated with cudaMallocHost,
 * which ensures that the memory is pinned and can be transferred
 * efficiently to the device.  Device memory is allocated with cudaMalloc.
 * After copying data from host to device, the kernel is launched to
 * compute C = A + B.  The result is copied back to the host, printed,
 * and finally all allocated resources are freed.  Importantly, the host
 * pinned memory is freed using cudaFreeHost, illustrating the correct
 * pair of allocation/deallocation functions.
 *
 * The code is self‑contained, compiles with nvcc, and includes error
 * checking for all CUDA API calls.  It also shows how to use cudaMemcpy
 * for both host‑to‑device and device‑to‑host transfers with pinned
 * memory.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel for vector addition
__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 20;          // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate pinned host memory
    float *h_A = nullptr;
    float *h_B = nullptr;
    float *h_C = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&h_A, size));
    CUDA_CHECK(cudaMallocHost((void**)&h_B, size));
    CUDA_CHECK(cudaMallocHost((void**)&h_C, size));

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify result (print first 5 elements)
    printf("First 5 results of A + B:\n");
    for (int i = 0; i < 5; ++i) {
        printf("h_C[%d] = %f\n", i, h_C[i]);
    }

    // Clean up device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Clean up pinned host memory
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    printf("Program completed successfully.\n");
    return 0;
}
