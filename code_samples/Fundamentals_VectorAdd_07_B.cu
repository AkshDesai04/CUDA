/*
Implement a kernel that computes the square root of every element: `output[i] = sqrtf(input[i])`.

Thinking process:
- The goal is to write a CUDA kernel that takes an input array of floats and writes the square root of each element to an output array.
- Use the device function sqrtf for single-precision floating point.
- The kernel should check bounds (idx < N) because the launch configuration might cover more threads than elements.
- Host code will allocate host and device memory, initialize the input array, copy it to device, launch the kernel, copy back results, and then free resources.
- For demonstration, use a fixed array size (e.g., 1024) and print first few results.
- Basic error handling can be omitted for brevity but should be added in production code.
- Include necessary headers: stdio.h, cuda_runtime.h, math.h.
- The kernel launch will use 256 threads per block, compute number of blocks accordingly.
- Compile with nvcc: nvcc -o sqrt_example sqrt_example.cu
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void sqrtKernel(const float* input, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        output[idx] = sqrtf(input[idx]);
    }
}

int main(void)
{
    const int N = 1024;
    size_t size = N * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // Initialize input array with some values
    for (int i = 0; i < N; ++i)
    {
        h_input[i] = (float)(i + 1);
    }

    float *d_input = NULL;
    float *d_output = NULL;

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sqrtKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print first 10 results
    printf("Input -> Output (sqrt):\n");
    for (int i = 0; i < 10 && i < N; ++i)
    {
        printf("%f -> %f\n", h_input[i], h_output[i]);
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
