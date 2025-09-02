```cuda
/*
Aim: Implement a kernel that squares every element in a vector: `output[i] = input[i] * input[i]`.

Thinking:
- We need a simple CUDA kernel that takes two pointers (input and output) and an integer size.
- Each thread will compute the index `i = blockIdx.x * blockDim.x + threadIdx.x`.
- If `i` is less than the vector size, we perform the squaring operation.
- In the host code, we allocate a small host vector, copy it to the device,
  launch the kernel, copy the result back, and print it to verify correctness.
- We'll use a single block dimension of 256 threads for simplicity.
- Memory allocation and copying will be done with `cudaMalloc` and `cudaMemcpy`.
- Basic error checking will be performed for each CUDA call.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void squareKernel(const float *input, float *output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float val = input[i];
        output[i] = val * val;
    }
}

int main(void)
{
    const int N = 10;
    size_t size = N * sizeof(float);

    // Host allocations
    float h_input[N];
    float h_output[N];

    // Initialize host input
    for (int i = 0; i < N; ++i)
        h_input[i] = (float)i + 1.0f; // 1.0, 2.0, ..., 10.0

    // Device allocations
    float *d_input = NULL;
    float *d_output = NULL;

    cudaError_t err;

    err = cudaMalloc((void**)&d_input, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc d_input failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaMalloc((void**)&d_output, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc d_output failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        return EXIT_FAILURE;
    }

    // Copy input data to device
    err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy HostToDevice failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return EXIT_FAILURE;
    }

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    squareKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return EXIT_FAILURE;
    }

    // Copy result back to host
    err = cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy DeviceToHost failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return EXIT_FAILURE;
    }

    // Print results
    printf("Input  : ");
    for (int i = 0; i < N; ++i)
        printf("%5.1f ", h_input[i]);
    printf("\n");

    printf("Output : ");
    for (int i = 0; i < N; ++i)
        printf("%5.1f ", h_output[i]);
    printf("\n");

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return EXIT_SUCCESS;
}
```