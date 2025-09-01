/*
Aim: Write a comment in your code showing the full `nvcc` command used to compile it.

Thinking:
1. The user explicitly requests that we embed a comment containing the full nvcc command that would compile this CUDA program.
2. The comment must appear somewhere in the .cu file. The easiest place is right before the main function, as a block comment or line comment.
3. The compile command itself can vary depending on target architecture, optimization level, etc. A typical command looks like: `nvcc -o vector_add vector_add.cu -arch=sm_70 -O2 -lineinfo`.
4. We'll choose a simple vector addition example to illustrate the kernel usage.
5. The code will consist of a `__global__` kernel, host allocation, copying, launch, and cleanup.
6. We'll keep it simple, but also robust: check for errors, free resources, etc.
7. The compile command comment will be a single line comment, starting with `//`, but could also be block comment. We'll use single line to keep it simple.
8. The final `.cu` file will have a multiline comment at the top with the aim and thoughts, then the code.
9. We need to ensure the `.cu` file is self-contained and can compile with the provided command.
10. We'll use the target architecture `sm_70` (CUDA 7.0) which is common. If you have a different device, you can modify `-arch` accordingly.
11. No additional text should be output; only the `.cu` file content.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Compile command: nvcc -o vector_add vector_add.cu -arch=sm_70 -O2 -lineinfo

// Simple vector addition kernel
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify a few results
    bool success = true;
    for (int i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            printf("Mismatch at index %d: got %f, expected %f\n", i, h_C[i], expected);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Vector addition successful! First 10 results verified.\n");
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
