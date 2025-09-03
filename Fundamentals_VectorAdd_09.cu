```cuda
/* 
Aim of the program:
Write a kernel that computes `D[i] = (A[i] + B[i]) * scalar`. This demonstrates passing multiple vectors and a scalar.

Thoughts and design decisions:
1. The program is a self‑contained CUDA application that allocates two input arrays (A and B),
   a scalar value, and an output array D on the device.
2. The kernel `addMulKernel` takes pointers to the input arrays, the output array,
   the scalar, and the number of elements `n`. Each thread computes one element of D
   if its global index is within bounds.
3. Host code:
   - Defines the size of the vectors (here 1024 for illustration).
   - Allocates host memory, initializes A and B with simple data.
   - Allocates device memory, copies inputs to device.
   - Determines a reasonable block and grid configuration.
   - Launches the kernel.
   - Copies the result back to host.
   - Prints a few sample results to verify correctness.
4. Error checking is performed after each CUDA runtime API call using a helper macro
   `CUDA_CHECK` to keep the code clean and robust.
5. The code uses single‑precision floats for simplicity; changing to double or another type
   would only require changing the data types.
6. The program is fully compilable with `nvcc` and does not depend on any external libraries beyond the CUDA runtime and standard C headers.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

__global__ void addMulKernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ D,
                             float scalar,
                             int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        D[i] = (A[i] + B[i]) * scalar;
    }
}

int main(void)
{
    const int N = 1024;                 // number of elements
    const size_t bytes = N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_D = (float*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;          // A[i] = i
        h_B[i] = (float)(N - i);    // B[i] = N - i
    }

    float scalar = 2.5f;              // example scalar

    // Device memory pointers
    float *d_A, *d_B, *d_D;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_D, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    addMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, scalar, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Sample results (index : D[i] = (A[i] + B[i]) * scalar):\n");
    for (int i = 0; i < 10; ++i) {
        printf("D[%d] = %f\n", i, h_D[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_D));
    free(h_A);
    free(h_B);
    free(h_D);

    return 0;
}
```