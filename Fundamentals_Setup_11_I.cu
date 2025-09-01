/*
Add the `-arch=sm_XX` flag to your `nvcc` command line to compile for a specific compute capability (e.g., `sm_75`).

Thinking:
- The aim is to remind the user that they should pass the `-arch=sm_XX` flag when invoking nvcc.
- The CUDA program itself doesn't need to change; it can be a simple kernel that demonstrates basic functionality.
- I will write a minimal program that allocates two arrays on the device, launches a kernel to add them element-wise, and copies the result back.
- The comment at the top will include the exact phrase requested and my reasoning for the implementation.
- The rest of the file will be standard CUDA C code, fully self‑contained.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple element‑wise addition kernel
__global__ void addVectors(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // Initialize input arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 1.0f;
        h_b[i] = (N - i) * 1.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify a few results
    bool ok = true;
    for (int i = 0; i < 10; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            ok = false;
            printf("Mismatch at %d: %f vs %f\n", i, h_c[i], expected);
        }
    }
    if (ok) printf("Vector addition succeeded.\n");

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
