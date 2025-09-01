```c
/* 
Aim of the program: Implement a kernel that computes x^n where n is passed as a parameter: 
output[i] = powf(input[i], n).

Thinking:
1. The requirement is to compute the power of each element in an input array using CUDA.
2. We will write a kernel that receives:
   - a pointer to the input array
   - a pointer to the output array
   - the exponent n (as an integer but cast to float for powf)
   - the size of the array so threads can check bounds.
3. Each thread will compute its global index and if it falls within the array size,
   it will apply `powf(input[idx], n)` and store the result in `output[idx]`.
4. In the host code, we will:
   - Create a small sample input array.
   - Allocate device memory for input and output.
   - Copy the input to device.
   - Launch the kernel with an appropriate grid/block configuration.
   - Copy the output back to host.
   - Print the results to verify correctness.
5. All CUDA error checks are omitted for brevity, but one could add them in production code.
6. The code is a complete, self‑contained CUDA program that compiles to a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Kernel to compute x^n for each element
__global__ void powKernel(const float *input, float *output, int n, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // Convert exponent to float for powf
        float exponent = (float)n;
        output[idx] = powf(input[idx], exponent);
    }
}

int main(void)
{
    const int size = 10;           // Number of elements
    const int exponent = 3;        // Compute cube of each element

    // Allocate host memory
    float h_input[size];
    float h_output[size];

    // Initialize input with sample values
    for (int i = 0; i < size; ++i)
    {
        h_input[i] = (float)(i + 1);  // 1.0, 2.0, ..., 10.0
    }

    // Device pointers
    float *d_input = NULL;
    float *d_output = NULL;

    // Allocate device memory
    cudaMalloc((void **)&d_input, size * sizeof(float));
    cudaMalloc((void **)&d_output, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Determine block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    powKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, exponent, size);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printf("Input\tPower(%d)\n", exponent);
    for (int i = 0; i < size; ++i)
    {
        printf("%6.2f\t%8.2f\n", h_input[i], h_output[i]);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
```