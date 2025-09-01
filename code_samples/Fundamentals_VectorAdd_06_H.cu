```cpp
/* 
Write a kernel for a generalized AXPY: Y[i] = a*X[i] + b*Y[i] + c, where a, b, c are scalars.

My thought process:
1. Understand the mathematical operation: for each element i of arrays X and Y, we want to compute Y[i] = a * X[i] + b * Y[i] + c.
2. We need a CUDA kernel that can be launched with enough threads to cover the array length. Each thread will handle one element.
3. The kernel signature will take the number of elements (N), device pointers to X and Y, and scalars a, b, c.
4. Inside the kernel, compute global thread index and guard against out-of-bounds.
5. Perform the computation: Y[idx] = a * X[idx] + b * Y[idx] + c.
6. On the host side, allocate and initialize host arrays, copy them to device, launch the kernel, copy the result back, and verify correctness.
7. Add basic error checking after CUDA calls for robustness.
8. Keep the code simple and self-contained so it can be compiled and run with nvcc. 
*/

#include <cstdio>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel for generalized AXPY
__global__ void generalized_axpy(int N, const float *X, float *Y,
                                 float a, float b, float c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Y[idx] = a * X[idx] + b * Y[idx] + c;
    }
}

// Host function to launch the kernel
void launch_generalized_axpy(int N, const float *h_X, float *h_Y,
                             float a, float b, float c)
{
    float *d_X = nullptr;
    float *d_Y = nullptr;

    size_t bytes = N * sizeof(float);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_X, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    generalized_axpy<<<blocksPerGrid, threadsPerBlock>>>(N, d_X, d_Y, a, b, c);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
}

int main()
{
    const int N = 1024;
    float h_X[N];
    float h_Y[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_X[i] = static_cast<float>(i) * 0.5f;      // X[i] = 0.5 * i
        h_Y[i] = static_cast<float>(i) * 1.0f;      // Y[i] = 1.0 * i
    }

    // Scalars
    float a = 2.0f;
    float b = 0.5f;
    float c = 1.0f;

    // Launch kernel via host helper
    launch_generalized_axpy(N, h_X, h_Y, a, b, c);

    // Verify result by printing first 10 elements
    printf("First 10 results of Y[i] = a*X[i] + b*Y[i] + c:\n");
    for (int i = 0; i < 10; ++i) {
        printf("Y[%d] = %f\n", i, h_Y[i]);
    }

    return 0;
}
```