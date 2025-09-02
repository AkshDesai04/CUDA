/*
Implement element-wise minimum.

Thinking:
- The program demonstrates the element-wise minimum operation on two arrays using CUDA.
- Host arrays `A` and `B` are initialized with sample values.
- Device memory for `A`, `B`, and the result `C` is allocated.
- The data from `A` and `B` are copied to the GPU.
- A simple kernel computes the minimum for each element and writes it to `C`.
- The result is copied back to the host and printed for verification.
- Error checking is performed on all CUDA API calls.
- Compile with: nvcc -o elem_min elem_min.cu
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel to compute element-wise minimum
__global__ void elementwiseMin(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = fminf(a[idx], b[idx]);
}

int main(void)
{
    const int N = 10;            // Number of elements
    const int SIZE = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(SIZE);
    float *h_B = (float*)malloc(SIZE);
    float *h_C = (float*)malloc(SIZE);

    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays with sample data
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)(i * 1.1);           // 0.0, 1.1, 2.2, ...
        h_B[i] = (float)(N - i);             // 10, 9, 8, ...
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, SIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_B, SIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_C, SIZE));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice));

    // Define execution configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    elementwiseMin<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, SIZE, cudaMemcpyDeviceToHost));

    // Print results
    printf("Index\tA\tB\tmin(A,B)\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d\t%.2f\t%.2f\t%.2f\n", i, h_A[i], h_B[i], h_C[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
