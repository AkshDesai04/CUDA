/*
Use `cudaPeekAtLastError()` immediately after the kernel launch to get the error code without waiting for a synchronization.

I thought about how to illustrate the use of cudaPeekAtLastError. The idea is to launch a simple kernel and then call cudaPeekAtLastError right after the launch, before any cudaDeviceSynchronize. This will give us the error code associated with the kernel launch itself, if any, without blocking the host. I decided to write a basic vector addition kernel, allocate memory, copy data, launch the kernel, peek at the error, and then synchronize and copy back the result. Additionally, to demonstrate that an error can be caught this way, I added a second kernel launch with intentionally bad launch parameters (setting block size too large for the device), then peek again. Finally, I print the error codes and messages to show the difference between a successful launch and an error.

The code is self-contained, compiles with nvcc, and demonstrates the use of cudaPeekAtLastError in practice. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10

// Simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

int main(void)
{
    float h_A[N], h_B[N], h_C[N];
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaError_t err;

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    // Allocate device memory
    err = cudaMalloc((void**)&d_A, N * sizeof(float));
    if (err != cudaSuccess) { printf("cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); return -1; }
    err = cudaMalloc((void**)&d_B, N * sizeof(float));
    if (err != cudaSuccess) { printf("cudaMalloc d_B error: %s\n", cudaGetErrorString(err)); return -1; }
    err = cudaMalloc((void**)&d_C, N * sizeof(float));
    if (err != cudaSuccess) { printf("cudaMalloc d_C error: %s\n", cudaGetErrorString(err)); return -1; }

    // Copy data to device
    err = cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy A error: %s\n", cudaGetErrorString(err)); return -1; }
    err = cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy B error: %s\n", cudaGetErrorString(err)); return -1; }

    // Launch kernel with proper parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Immediately after launch, use cudaPeekAtLastError to capture any launch error
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("First launch error (should be none): %s\n", cudaGetErrorString(err));
    } else {
        printf("First launch succeeded (no error).\n");
    }

    // Synchronize to ensure kernel has finished before copying back
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err)); return -1; }

    // Copy result back to host
    err = cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("cudaMemcpy C error: %s\n", cudaGetErrorString(err)); return -1; }

    // Print result
    printf("Result of vector addition:\n");
    for (int i = 0; i < N; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Now intentionally launch with an invalid configuration (too many threads per block)
    int badThreadsPerBlock = 32768; // Likely exceeds device capability
    vectorAdd<<<blocksPerGrid, badThreadsPerBlock>>>(d_A, d_B, d_C, N);

    // Peek error immediately
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("Second launch error (expected due to bad launch config): %s\n", cudaGetErrorString(err));
    } else {
        printf("Second launch succeeded (unexpected).\n");
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
