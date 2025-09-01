/*
Aim of the program:
Change the data type from int to float and initialize the host array with floating-point values.

Thought process:
The original program (not shown) likely performed a simple CUDA operation such as vector addition using int arrays. To meet the requirement, we need to:
1. Replace all `int` types with `float` throughout the host and device code.
2. Initialize the host arrays with floating-point literals (e.g., 1.0f, 2.5f, etc.).
3. Ensure kernel and memory operations use `float` pointers.
4. Keep the structure of a typical CUDA example: allocate device memory, copy host to device, launch kernel, copy back, and print results.
5. Include proper error checking for CUDA API calls for robustness.

The resulting code demonstrates vector addition with floating-point data, matching the requested changes.

*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    const int N = 5;                     // number of elements
    const int SIZE = N * sizeof(float);  // size in bytes

    // Allocate host memory and initialize with floating-point values
    float h_a[N] = {1.0f, 2.0f, 3.5f, 4.5f, 5.0f};
    float h_b[N] = {5.0f, 4.0f, 3.0f, 2.5f, 1.5f};
    float h_c[N] = {0.0f};

    // Allocate device memory
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    cudaError_t err = cudaMalloc((void**)&d_a, SIZE);
    if (err != cudaSuccess) { printf("cudaMalloc d_a failed\n"); return -1; }
    err = cudaMalloc((void**)&d_b, SIZE);
    if (err != cudaSuccess) { printf("cudaMalloc d_b failed\n"); return -1; }
    err = cudaMalloc((void**)&d_c, SIZE);
    if (err != cudaSuccess) { printf("cudaMalloc d_c failed\n"); return -1; }

    // Copy host data to device
    cudaMemcpy(d_a, h_a, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, SIZE, cudaMemcpyHostToDevice);

    // Launch kernel: one thread per element
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, SIZE, cudaMemcpyDeviceToHost);

    // Print results
    printf("Vector addition result:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("h_c[%d] = %f\n", i, h_c[i]);
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
