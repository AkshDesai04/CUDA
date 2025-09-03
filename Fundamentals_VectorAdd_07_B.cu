/*
Aim of the program: Implement a kernel that computes the square root of every element: `output[i] = sqrtf(input[i])`.

Thinking:
- The goal is to provide a simple CUDA example that demonstrates launching a kernel to compute the square root of each element in an array.
- We need a kernel `sqrtKernel` that takes pointers to input and output arrays and the number of elements.
- Each thread computes one element if its global index is within bounds.
- On the host, allocate and initialize a host input array (e.g., a sequence of positive numbers).
- Allocate device memory for input and output, copy the input to device.
- Determine a suitable grid/block configuration (e.g., 256 threads per block).
- Launch the kernel and copy the result back to the host.
- Verify the result by printing a few values.
- Include basic CUDA error checking for robustness.
- Use `sqrtf` for single-precision square root.
*/

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// Simple macro for error checking
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                  \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel that computes sqrtf for each element
__global__ void sqrtKernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        output[idx] = sqrtf(input[idx]);
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_input  = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input array with some values
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(i + 1); // avoid zero to keep sqrt meaningful
    }

    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc((void**)&d_input,  size));
    CHECK_CUDA(cudaMalloc((void**)&d_output, size));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    sqrtKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    CHECK_CUDA(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("First 10 results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("sqrtf(%f) = %f\n", h_input[i], h_output[i]);
    }

    // Clean up
    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return EXIT_SUCCESS;
}
