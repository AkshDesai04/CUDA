```cuda
/*
Implement the canonical first parallel program: element-wise vector addition.
The program demonstrates a simple CUDA application that performs vector addition on the GPU.
We declare three vectors A, B, and C of length 256. A and B are initialized on the host.
Memory for these vectors is allocated on the device, and A and B are copied to device memory.
A kernel with a single block of 256 threads is launched. Each thread processes a single
index i, computing C[i] = A[i] + B[i]. After the kernel execution, the resulting vector C
is copied back to host memory. The program then prints the resulting vector to verify
correctness. This showcases the basics of CUDA memory management, kernel launch,
and synchronization.
*/
#include <stdio.h>
#include <cuda.h>

#define N 256

// CUDA kernel for element-wise vector addition
__global__ void vecAdd(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    float h_A[N], h_B[N], h_C[N];
    float *d_A, *d_B, *d_C;

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel: one block of 256 threads
    dim3 blockSize(N);
    dim3 gridSize(1);
    vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    printf("Result of vector addition:\n");
    for (int i = 0; i < N; ++i) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```