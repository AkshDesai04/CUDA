```c
/* 
Aim of the program:
Implement a kernel that squares every element in a vector: `output[i] = input[i] * input[i]`.

Thought process:
1. Define a simple CUDA kernel that takes two pointers, input and output, and an integer size.
   Each thread will compute the index `idx = blockIdx.x * blockDim.x + threadIdx.x`.
   If `idx` is within bounds, it will read `input[idx]`, square it, and write to `output[idx]`.

2. On the host side:
   - Allocate an array of floats on the device for input and output.
   - Allocate a corresponding array on the host to hold the result.
   - Initialize the host input array with sample data (e.g., 0.0, 1.0, 2.0, ...).
   - Copy the host input to the device.
   - Determine a suitable block and grid size (e.g., 256 threads per block).
   - Launch the kernel.
   - Copy the device output back to the host.
   - Verify the result by printing a few elements.

3. Include basic error checking after CUDA API calls for robustness.
4. Keep the code minimal but functional; it can be compiled with `nvcc -o square_vector square_vector.cu` and run.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel that squares each element of the input vector
__global__ void squareKernel(const float* input, float* output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float val = input[idx];
        output[idx] = val * val;
    }
}

// Helper function for checking CUDA errors
static inline void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    const int N = 1024;                     // Number of elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_input  = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    if (!h_input || !h_output)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host input array
    for (int i = 0; i < N; ++i)
    {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_input, *d_output;
    checkCudaError(cudaMalloc((void**)&d_input,  size), "cudaMalloc d_input");
    checkCudaError(cudaMalloc((void**)&d_output, size), "cudaMalloc d_output");

    // Copy input from host to device
    checkCudaError(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    squareKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    // Check for launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    // Verify a few results
    printf("Input -> Output (first 10 elements):\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("%f squared = %f\n", h_input[i], h_output[i]);
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
```