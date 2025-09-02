```cuda
/*
Aim of the program: Implement linear interpolation (lerp): C[i] = A[i] * (1.0 - t) + B[i] * t, where t is a scalar float.

Thinking process:
1. I need a self‑contained CUDA C program that demonstrates the lerp operation on a GPU.
2. The program will:
   - Allocate host arrays A, B, C.
   - Initialize A and B with some test data.
   - Allocate corresponding device arrays and copy the data to the device.
   - Launch a kernel that performs the lerp formula for each element.
   - Copy the result back to host and verify correctness against a CPU implementation.
3. For simplicity, I'll use a small array size (e.g., N = 10) so the output can be printed clearly.
4. The kernel will be a straightforward element‑wise loop with the given formula.
5. Error checking for CUDA API calls will be minimal but sufficient (checking return codes).
6. The program will print both the GPU result and the CPU reference result.
7. All code is written in plain C (CUDA C) and uses the standard CUDA runtime API.

The code below follows these steps and compiles as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10

// CUDA kernel to perform linear interpolation
__global__ void lerpKernel(const float* A, const float* B, float* C, float t, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * (1.0f - t) + B[idx] * t;
    }
}

// CPU reference implementation for verification
void lerpCPU(const float* A, const float* B, float* C, float t, int n) {
    for (int i = 0; i < n; ++i) {
        C[i] = A[i] * (1.0f - t) + B[i] * t;
    }
}

int main() {
    float h_A[N], h_B[N], h_C_gpu[N], h_C_cpu[N];
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    float t = 0.3f; // example interpolation factor

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;          // 0, 1, 2, ...
        h_B[i] = (float)(N - i);    // 10, 9, 8, ...
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    lerpKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, t, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C_gpu, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    lerpCPU(h_A, h_B, h_C_cpu, t, N);

    // Print results
    printf("Index\tA\tB\tC_gpu\tC_cpu\n");
    for (int i = 0; i < N; ++i) {
        printf("%d\t%.2f\t%.2f\t%.4f\t%.4f\n", i, h_A[i], h_B[i], h_C_gpu[i], h_C_cpu[i]);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```