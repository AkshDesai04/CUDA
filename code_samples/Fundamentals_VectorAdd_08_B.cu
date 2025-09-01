```c
/*
Implement scalar multiplication: `output[i] = input[i] * scalar_value`.

The goal is to demonstrate a simple CUDA program that multiplies each element of an input array by a scalar value and stores the result in an output array. The approach involves:

1. Defining a CUDA kernel `scalarMul` that performs the element‑wise multiplication. Each thread processes one element, using its global index to access the input and write to the output array.
2. In the host code, allocating host memory for the input and output arrays, initializing the input with sample values.
3. Allocating corresponding device memory with `cudaMalloc`, copying the input from host to device.
4. Launching the kernel with a configuration of blocks and threads that covers all elements.
5. Copying the output array back from device to host.
6. Printing the results to verify correctness.
7. Cleaning up all allocated memory and checking for CUDA errors after each CUDA call.

Key CUDA concepts used:
- Kernel launch syntax: `kernel<<<grid, block>>>(...)`
- Calculating global thread index: `int idx = blockIdx.x * blockDim.x + threadIdx.x;`
- Bounds checking inside the kernel to avoid out‑of‑bounds memory access.
- Basic error handling with a helper macro `CUDA_CHECK`.

This minimal example is self‑contained and can be compiled with `nvcc` and run on a system with an appropriate CUDA‑capable GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                     \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));      \
            return EXIT_FAILURE;                                                  \
        }                                                                         \
    } while (0)

// CUDA kernel for scalar multiplication
__global__ void scalarMul(const float *input, float *output, float scalar, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * scalar;
    }
}

int main(void)
{
    const int N = 1 << 20;          // Number of elements (1M)
    const size_t size = N * sizeof(float);
    const float scalar_value = 3.14f;

    // Host allocations
    float *h_input  = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)i;
    }

    // Device allocations
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void **)&d_input,  size));
    CUDA_CHECK(cudaMalloc((void **)&d_output, size));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    scalarMul<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, scalar_value, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Verify results (print first 10 elements)
    printf("First 10 results of scalar multiplication (scalar = %f):\n", scalar_value);
    for (int i = 0; i < 10; ++i) {
        printf("output[%d] = %f\n", i, h_output[i]);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    printf("Scalar multiplication completed successfully.\n");
    return EXIT_SUCCESS;
}
```