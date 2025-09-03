```cuda
/*
Aim of the program: Explain in a comment that because kernel launches are asynchronous, the CPU can continue working on other tasks while the GPU is busy.

My thinking:
- CUDA kernel launches are non-blocking by default. When the host (CPU) calls a kernel, control returns immediately, allowing the host to perform other operations while the device (GPU) executes the kernel in the background.
- To demonstrate this, the program will launch a simple vector addition kernel on the GPU.
- After launching the kernel, the CPU will enter a loop that simulates additional work (e.g., printing a message or performing a small computation).
- During this time, the GPU continues processing the kernel.
- Finally, the host will synchronize with the device to ensure the kernel has finished before accessing the results.
- This illustrates the asynchrony: the CPU does not wait for the kernel to complete before moving on to other tasks.

The comment inside the code will explicitly state that kernel launches are asynchronous and explain the consequence.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that adds two vectors
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    const int N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = i * 0.5f;
        h_B[i] = i * 1.5f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch kernel (asynchronous)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // While the GPU is working on the kernel, the CPU continues executing this loop
    // This demonstrates that kernel launches are asynchronous
    for (int i = 0; i < 5; ++i) {
        printf("CPU is doing other work... iteration %d\n", i);
        // Simulate some CPU work
        float dummy = 0.0f;
        for (int j = 0; j < 1000000; ++j) {
            dummy += j * 0.000001f;
        }
    }

    // Synchronize to ensure kernel completion before accessing results
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify a few results
    for (int i = 0; i < 5; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Clean up
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
```