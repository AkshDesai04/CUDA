/*
Use an array in constant memory, e.g., `__constant__ float coefficients[16];`. Copy 16 floats from the host and use them in a kernel.

Thinking:
- We need to create a constant memory array of 16 floats and copy data into it from the host.
- Define a simple kernel that uses those coefficients to perform per-element multiplication with an input array.
- Allocate input and output arrays on the host, initialize input, copy input to device global memory.
- Copy coefficients to the constant memory region using cudaMemcpyToSymbol.
- Launch a single block of 16 threads (one per element).
- After kernel execution, copy results back to the host and print them.
- Include basic error checking and resource cleanup.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Constant memory array
__constant__ float coefficients[16];

// Simple kernel that multiplies each input element by its corresponding coefficient
__global__ void multiplyKernel(const float *input, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 16) {
        output[idx] = input[idx] * coefficients[idx];
    }
}

// Macro for error checking
#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

int main(void) {
    // Host arrays
    float h_input[16];
    float h_output[16];
    float h_coefficients[16];

    // Initialize input and coefficient arrays
    for (int i = 0; i < 16; ++i) {
        h_input[i] = (float)(i + 1);          // 1.0, 2.0, ..., 16.0
        h_coefficients[i] = (float)(i + 0.5f); // 0.5, 1.5, ..., 15.5
    }

    // Device pointers
    float *d_input = nullptr;
    float *d_output = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_input, 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_output, 16 * sizeof(float)));

    // Copy input array to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, 16 * sizeof(float), cudaMemcpyHostToDevice));

    // Copy coefficient array to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(coefficients, h_coefficients,
                                 16 * sizeof(float), 0, cudaMemcpyHostToDevice));

    // Launch kernel: one block of 16 threads
    multiplyKernel<<<1, 16>>>(d_input, d_output);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 16 * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    printf("Result:\n");
    for (int i = 0; i < 16; ++i) {
        printf("h_input[%2d] * coefficients[%2d] = %8.4f\n",
               i, i, h_output[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
