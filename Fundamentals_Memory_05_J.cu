/*
Aim of the program which is passed in this prompt word to word:
Forget to call `cudaFree`. Run the program. This is a memory leak.

My entire thinking:
- The requirement is to produce a CUDA C program that deliberately does not call cudaFree,
  thereby causing a memory leak on the device.
- To make it a complete program, I will allocate device memory, launch a simple kernel that
  performs element-wise addition of two arrays, copy the result back to host, print a few
  values for verification, and then exit without freeing the device memory.
- I will include basic error checking for CUDA API calls, but I will purposely omit the
  cudaFree calls. Host memory will be freed normally; the focus is on device memory leak.
- The code will compile as a .cu file and can be run with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel to add two arrays
__global__ void addArrays(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 1.0f;
        h_b[i] = (N - i) * 1.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaError_t err;

    err = cudaMalloc((void **)&d_a, bytes);
    if (err != cudaSuccess) { printf("cudaMalloc d_a failed\n"); return 1; }
    err = cudaMalloc((void **)&d_b, bytes);
    if (err != cudaSuccess) { printf("cudaMalloc d_b failed\n"); return 1; }
    err = cudaMalloc((void **)&d_c, bytes);
    if (err != cudaSuccess) { printf("cudaMalloc d_c failed\n"); return 1; }

    // Copy data to device
    err = cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy to d_a failed\n"); return 1; }
    err = cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy to d_b failed\n"); return 1; }

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    addArrays<<<blocks, threads>>>(d_a, d_b, d_c, N);

    // Copy result back
    err = cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("cudaMemcpy from d_c failed\n"); return 1; }

    // Verify a few results
    printf("Sample results:\n");
    for (int i = 0; i < 5; ++i)
        printf("c[%d] = %f\n", i, h_c[i]);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    // Intentionally forget to free device memory: cudaFree(d_a), cudaFree(d_b), cudaFree(d_c)
    // This results in a device memory leak.

    return 0;
}
