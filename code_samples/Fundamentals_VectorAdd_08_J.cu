/*
Implement `output[i] = max(input[i], scalar_value)`.

The aim of this program is to compute an element‑wise maximum between a 1‑D array of floating point numbers (`input`) and a single scalar value (`scalar_value`).  
The operation is performed on the GPU using a CUDA kernel, which iterates over the array indices and writes the maximum of the two operands to the `output` array.  The program demonstrates:
1. Allocation of device memory for input and output arrays.
2. Transfer of data between host and device.
3. Launch of a CUDA kernel with a configurable block size and grid size.
4. Copying the result back to the host and printing a few values for verification.
5. Basic error checking after CUDA API calls.

Design decisions:
- The kernel uses a simple one‑dimensional index mapping: each thread processes one element of the array.  This is efficient for contiguous memory and easy to understand.
- The scalar value is passed as a kernel argument; no need to use constant memory or textures.
- The kernel launch configuration is chosen such that each block contains 256 threads, a common choice that balances occupancy and resource usage.
- The host code checks that the size of the array is a multiple of the block size by adjusting the grid size accordingly.
- Error handling is implemented with a helper macro `CUDA_CHECK` that aborts execution and prints the CUDA error if any API call fails.

The program can be compiled with `nvcc -o max_scalar max_scalar.cu` and run as `./max_scalar`.  It will output the first few elements of the input and the corresponding output to demonstrate the operation.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel that computes output[i] = max(input[i], scalar_value)
__global__ void max_scalar_kernel(const float* input, float* output, float scalar_value, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float val = input[idx];
        output[idx] = (val > scalar_value) ? val : scalar_value;
    }
}

int main(void)
{
    const size_t N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);
    const float scalar_value = 3.14f;

    // Allocate host memory
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input with random values
    for (size_t i = 0; i < N; ++i) {
        h_input[i] = (float)rand() / RAND_MAX * 10.0f; // values in [0,10]
    }

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input, size));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    max_scalar_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, scalar_value, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Verify result by printing first 10 elements
    printf("Index\tInput\t\tOutput\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d\t%f\t%f\n", i, h_input[i], h_output[i]);
    }

    // Cleanup
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return EXIT_SUCCESS;
}
