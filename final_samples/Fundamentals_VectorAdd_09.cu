```c
/*
Write a kernel that computes `D[i] = (A[i] + B[i]) * scalar`. This demonstrates passing multiple vectors and a scalar.

Thinking process:
- The goal is to create a CUDA program that takes two input vectors A and B, adds them element-wise, multiplies each sum by a scalar, and stores the result in vector D.
- The kernel will have parameters for the input pointers, the output pointer, the scalar value, and the number of elements to process.
- Each thread will handle one element: it will read A[i] and B[i], compute the sum, multiply by the scalar, and write the result to D[i].
- In the host code, allocate memory on the device for A, B, and D, copy input data from host to device, launch the kernel with an appropriate grid/block configuration, copy the result back, and finally verify correctness.
- We'll include basic error checking via a macro to keep the code clean.
- For demonstration, we’ll initialize A and B with sequential numbers and use a scalar of 2.5, and print the first few results.
- The program will compile as a single .cu file. Use `nvcc` to compile.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that computes D[i] = (A[i] + B[i]) * scalar
__global__ void addAndScaleKernel(const float *A, const float *B, float *D, float scalar, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        D[idx] = (A[idx] + B[idx]) * scalar;
    }
}

int main(void)
{
    const size_t N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);
    const float scalar = 2.5f;

    // Host allocations
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_D = (float *)malloc(bytes);

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Device allocations
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_D = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_D, bytes));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addAndScaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, scalar, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Sample results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("D[%d] = (A[%d] + B[%d]) * %.2f = %.2f\n", i, i, i, scalar, h_D[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_D);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_D));

    return 0;
}
```