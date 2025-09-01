```cpp
/*
Implement `C[i] = abs(A[i] - B[i])`.

Thinking:
- The program should be a complete CUDA C++ program that compiles into a .cu file.
- It will define a kernel that takes three integer arrays (A, B, C) and the number of elements N.
- Each thread will compute one element: C[i] = abs(A[i] - B[i]).
- Host code will allocate host arrays, initialize them, allocate device memory, copy data to device,
  launch the kernel, copy results back, and print them for verification.
- Basic CUDA error checking will be performed after each CUDA API call.
- Block and grid sizes will be chosen to cover all N elements.
- The program will use `int` for simplicity; `abs` is the C standard library function.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h> // for abs

// CUDA error checking macro
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel to compute absolute difference element-wise
__global__ void absDiffKernel(const int *A, const int *B, int *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = abs(A[idx] - B[idx]);
    }
}

int main(void)
{
    const int N = 1024;            // Number of elements
    const int bytes = N * sizeof(int);

    // Allocate host memory
    int *h_A = (int*)malloc(bytes);
    int *h_B = (int*)malloc(bytes);
    int *h_C = (int*)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host arrays.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays with sample data
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;                     // 0, 1, 2, ...
        h_B[i] = N - i;                 // N, N-1, N-2, ...
    }

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Determine grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    absDiffKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify results (print first 10 elements for brevity)
    printf("First 10 results of C[i] = abs(A[i] - B[i]):\n");
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("C[%d] = %d\n", i, h_C[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}
```