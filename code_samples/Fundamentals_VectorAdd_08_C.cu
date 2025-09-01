```cuda
/*
Implement scalar division: `output[i] = input[i] / scalar_value`.

Thinking:
The task is to write a simple CUDA program that performs element-wise scalar division on an array.
We will:
1. Define a CUDA kernel that divides each element of the input array by a scalar.
2. In the host code, create a small input array, allocate device memory, copy data to the device,
   launch the kernel, copy the result back, and print a few elements to verify correctness.
3. Use a typical error-checking macro to catch CUDA API errors.
4. Keep the example self‑contained, compiling with `nvcc` and running on any CUDA device.

The kernel will compute the global thread index and, if within bounds, perform the division.
We will use a block size of 256 threads and compute the number of blocks accordingly.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel to perform scalar division
__global__ void scalar_divide(const float *input, float *output, float scalar, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] / scalar;
    }
}

int main(void) {
    const int N = 1024;            // Number of elements
    const float scalar_value = 2.5f; // Scalar divisor

    // Host allocation
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    if (h_input == NULL || h_output == NULL) {
        fprintf(stderr, "Failed to allocate host arrays.\n");
        return EXIT_FAILURE;
    }

    // Initialize input with some values
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)i + 1.0f; // 1.0, 2.0, 3.0, ...
    }

    // Device allocation
    float *d_input = NULL;
    float *d_output = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    scalar_divide<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, scalar_value, N);
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("First 10 results of scalar division by %f:\n", scalar_value);
    for (int i = 0; i < 10; ++i) {
        printf("h_input[%d] = %f, h_output[%d] = %f\n", i, h_input[i], i, h_output[i]);
    }

    // Clean up
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    // Reset device before exiting
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
```