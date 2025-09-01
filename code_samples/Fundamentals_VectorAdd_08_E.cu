/*
Implement `output[i] = scalar_value - input[i]`.

Thought process:
- The task is to perform an element‑wise operation on an array: subtract each element of an input array from a given scalar and store the result in an output array.
- In CUDA, this operation is naturally expressed as a kernel where each thread handles one element of the arrays.
- We'll write a kernel `subtract_scalar` that takes pointers to the input and output arrays, the scalar value, and the number of elements `N`. Each thread computes its global index `idx` and, if `idx < N`, performs `output[idx] = scalar - input[idx]`.
- The host code will:
  1. Define the number of elements `N` and allocate host arrays for `input` and `output`.
  2. Initialize `input` with some test data (e.g., sequential values) and set a scalar value.
  3. Allocate device memory for the input and output arrays.
  4. Copy the input data from host to device.
  5. Launch the kernel with an appropriate grid and block configuration.
  6. Copy the result back from device to host.
  7. Print a few results to verify correctness.
  8. Clean up all allocated memory.
- For simplicity and clarity, error checking is minimal but sufficient for this example. The code uses `float` for data types, but it can be adapted to other numeric types if needed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void subtract_scalar(const float *input, float *output, float scalar, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        output[idx] = scalar - input[idx];
    }
}

int main(void)
{
    const int N = 1024;               // Number of elements
    const int size = N * sizeof(float);

    // Allocate host memory
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    if (h_input == NULL || h_output == NULL)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return 1;
    }

    // Initialize input array with some values and scalar
    float scalar = 10.0f;
    for (int i = 0; i < N; ++i)
    {
        h_input[i] = (float)i;  // example data: 0, 1, 2, ...
    }

    // Allocate device memory
    float *d_input = NULL;
    float *d_output = NULL;
    cudaError_t err = cudaMalloc((void **)&d_input, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc for d_input failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc((void **)&d_output, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc for d_output failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        return 1;
    }

    // Copy input data from host to device
    err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    subtract_scalar<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, scalar, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }

    // Verify and print first 10 results
    printf("First 10 results of scalar - input:\n");
    for (int i = 0; i < 10 && i < N; ++i)
    {
        printf("output[%d] = %.2f - %.2f = %.2f\n", i, scalar, h_input[i], h_output[i]);
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
