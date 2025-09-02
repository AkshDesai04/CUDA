```cuda
/*
Implement scalar division: `output[i] = input[i] / scalar_value`.

Thinking:
- The goal is to create a simple CUDA program that demonstrates scalar division of an array of
  floating-point numbers on the GPU.
- We will write a CUDA kernel that receives a pointer to the input array, a pointer to the output
  array, the number of elements, and the scalar divisor.
- Each thread will compute one element of the output array, performing the division if its
  global index is within bounds.
- In the host code we will:
  1. Allocate and initialize a host input array.
  2. Allocate device memory for input and output arrays.
  3. Copy the input array from host to device.
  4. Launch the kernel with an appropriate grid/block configuration.
  5. Copy the result back to the host.
  6. Print a few elements to verify correctness.
  7. Clean up device and host memory.
- We will use a simple error-checking macro to catch CUDA API errors.
- The scalar divisor will be a float, e.g., 2.0f, so the output should be half of the input.
- We keep the program self‑contained so it can be compiled with `nvcc` and run on any CUDA
  capable device.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel that divides each element of the input array by a scalar value.
__global__ void scalar_divide(const float* input, float* output, int n, float scalar_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] / scalar_value;
    }
}

int main() {
    const int n = 1024;                // Number of elements
    const float scalar_value = 2.0f;   // Divisor

    // Allocate host memory
    float* h_input  = new float[n];
    float* h_output = new float[n];

    // Initialize host input array with some values
    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(i) * 1.0f; // e.g., 0.0, 1.0, 2.0, ...
    }

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, n * sizeof(float)));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    scalar_divide<<<blocks, threads_per_block>>>(d_input, d_output, n, scalar_value);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy the result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("First 10 elements after division by %.2f:\n", scalar_value);
    for (int i = 0; i < 10; ++i) {
        printf("h_output[%d] = %f\n", i, h_output[i]);
    }

    // Clean up
    delete[] h_input;
    delete[] h_output;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
```