/*
 Aim: Implement scalar multiplication: output[i] = input[i] * scalar_value.

 Thinking:
 1. We need a CUDA kernel that performs element‑wise multiplication of an input array by a scalar.
 2. The kernel will compute the global thread index and apply the multiplication only if the index is within bounds.
 3. The host code will allocate memory on both host and device, initialize the input array, copy it to the device,
    launch the kernel with an appropriate grid/block configuration, copy the result back, and optionally verify it.
 4. Basic CUDA error checking macros are included for robustness.
 5. The code is self‑contained and can be compiled with nvcc (e.g., `nvcc scalar_mul.cu -o scalar_mul`).
 6. For demonstration, we use a modest array size and a simple scalar value; the user can change these as needed.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

/* Kernel: multiply each element of input by scalar and store in output */
__global__ void scalarMul(const float *input, float *output, float scalar, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * scalar;
    }
}

int main(void)
{
    const int N = 1 << 20;          // Number of elements (about 1M)
    const size_t bytes = N * sizeof(float);
    const float scalar = 2.5f;      // Example scalar value

    // Allocate host memory
    float *h_input  = (float *)malloc(bytes);
    float *h_output = (float *)malloc(bytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input with some values (e.g., sequential numbers)
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_input = NULL;
    float *d_output = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_input, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_output, bytes));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Determine launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    scalarMul<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, scalar, N);
    CHECK_CUDA(cudaGetLastError());          // Check for launch errors
    CHECK_CUDA(cudaDeviceSynchronize());     // Wait for kernel to finish

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 elements
    printf("First 10 results (input * %f):\n", scalar);
    for (int i = 0; i < 10; ++i) {
        printf("h_output[%d] = %f\n", i, h_output[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);

    printf("Scalar multiplication completed successfully.\n");
    return EXIT_SUCCESS;
}
