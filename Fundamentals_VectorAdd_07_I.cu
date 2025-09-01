```cpp
/*
Aim of the program: Implement `output[i] = sinf(input[i])`.

Thinking process:
1. The task requires a CUDA C/C++ program that applies the sine function to each element
   of an input array and stores the result in an output array. 
2. CUDA kernels are launched with a grid of blocks and threads; each thread processes
   one element index `i`. We need to guard against indices beyond the array size.
3. The function `sinf` from the math library works on `float`. We will use it in the
   kernel.
4. We will write a simple `main` that:
   - allocates host arrays,
   - initializes the input with some test values,
   - allocates device memory,
   - copies data from host to device,
   - launches the kernel,
   - copies the result back to host,
   - prints a few results for verification.
5. Memory cleanup is performed at the end. We use `cudaMalloc`, `cudaMemcpy`, and
   `cudaFree`. The program will compile with `nvcc` and can be run on any CUDA
   compatible device.
6. The code will be self-contained, with no external dependencies beyond the CUDA
   runtime and standard C++ headers.

*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// CUDA kernel that computes sinf for each element
__global__ void sinf_kernel(const float* __restrict__ d_input,
                            float* __restrict__ d_output,
                            int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        d_output[idx] = sinf(d_input[idx]);
    }
}

int main(void)
{
    const int N = 10;                // number of elements
    const int bytes = N * sizeof(float);

    // Allocate and initialize host memory
    float h_input[N];
    float h_output[N];

    // Fill input with sample values (e.g., 0 to 1.5π)
    for (int i = 0; i < N; ++i)
    {
        h_input[i] = i * (M_PI / 4.0f); // multiples of π/4
    }

    // Allocate device memory
    float *d_input = nullptr;
    float *d_output = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_input, bytes);
    if (err != cudaSuccess) { printf("cudaMalloc d_input failed: %s\n", cudaGetErrorString(err)); return -1; }
    err = cudaMalloc((void**)&d_output, bytes);
    if (err != cudaSuccess) { printf("cudaMalloc d_output failed: %s\n", cudaGetErrorString(err)); cudaFree(d_input); return -1; }

    // Copy input from host to device
    err = cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy to device failed: %s\n", cudaGetErrorString(err)); cudaFree(d_input); cudaFree(d_output); return -1; }

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    sinf_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); cudaFree(d_input); cudaFree(d_output); return -1; }

    // Copy result from device to host
    err = cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("cudaMemcpy to host failed: %s\n", cudaGetErrorString(err)); cudaFree(d_input); cudaFree(d_output); return -1; }

    // Print results
    printf("Input\t\tOutput (sinf)\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%f\t%f\n", h_input[i], h_output[i]);
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
```