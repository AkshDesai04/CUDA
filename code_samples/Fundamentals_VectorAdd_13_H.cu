/*
Implement a clamp function: `C[i] = min(max(A[i], min_val), max_val)`.

In this program, we will demonstrate how to implement a clamp operation on the GPU using CUDA. The clamp operation is a common element-wise operation where each input value is constrained to lie between a specified minimum and maximum. The formula `C[i] = min(max(A[i], min_val), max_val)` first raises the input `A[i]` to `min_val` if it is smaller, then lowers it to `max_val` if it is larger, otherwise leaves it unchanged.

We will write a CUDA kernel that performs this operation on an array of floats. The kernel will be launched with enough threads to cover all array elements. After launching, we will copy the results back to the host and verify a few entries. The code also includes basic error checking to ensure memory allocations and kernel launches succeed.

The program includes:
- Host-side memory allocation and initialization.
- Device memory allocation.
- Kernel definition and launch.
- Copying data back to host and printing results.
- Cleanup of resources.

This example can be adapted for any numeric type (int, double, etc.) by adjusting the kernel accordingly. For simplicity, this implementation uses `float` and a small fixed-size array for demonstration.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",      \
                    #call, __FILE__, __LINE__,                    \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                          \
    } while (0)

__global__ void clampKernel(float *C, const float *A, int N,
                            float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = A[idx];
        // Clamp: first max with min_val, then min with max_val
        val = fmaxf(val, min_val);
        val = fminf(val, max_val);
        C[idx] = val;
    }
}

int main(void) {
    const int N = 1024;          // Number of elements
    const float min_val = 0.0f;  // Minimum clamp value
    const float max_val = 10.0f; // Maximum clamp value

    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (!h_A || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input array with random values between -10 and 20
    for (int i = 0; i < N; ++i) {
        h_A[i] = -10.0f + static_cast<float>(rand()) / RAND_MAX * 30.0f;
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_C = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    clampKernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, N, min_val, max_val);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA(cudaDeviceSynchronize()); // Ensure kernel completion

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Print first 10 elements for verification
    printf("Index\tA\tC (clamped)\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d\t%.4f\t%.4f\n", i, h_A[i], h_C[i]);
    }

    // Free resources
    free(h_A);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_C));

    // Reset device
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
