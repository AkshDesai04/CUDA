```c
/*
Aim of the program: Write a kernel that computes `D[i] = (A[i] + B[i]) * scalar`. This demonstrates passing multiple vectors and a scalar.

Thinking:
1. The user wants a CUDA kernel that takes two input arrays A and B, a scalar, and writes to an output array D.
2. The kernel should be simple, using a single loop over indices and a basic arithmetic operation.
3. We need to provide a complete .cu file, including host code that allocates memory, copies data, launches the kernel, and retrieves results.
4. Use a small example size (e.g., N = 1024) for demonstration.
5. Include basic error checking for CUDA API calls to help debugging.
6. Use `float` for data type; scalar also `float`.
7. Kernel signature: `__global__ void vecAddMul(const float *A, const float *B, float *D, float scalar, int N)`.
8. Index calculation: `int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx < N) D[idx] = (A[idx] + B[idx]) * scalar;`
9. Launch configuration: use `blockSize = 256` threads per block, `gridSize = (N + blockSize - 1) / blockSize`.
10. After kernel execution, copy D back to host and print first few values to confirm.
11. Add comments in code for clarity.
12. The code must compile with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel that computes D[i] = (A[i] + B[i]) * scalar
__global__ void vecAddMul(const float *A, const float *B, float *D, float scalar, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        D[idx] = (A[idx] + B[idx]) * scalar;
    }
}

int main(void) {
    const int N = 1024;               // Number of elements
    const size_t size = N * sizeof(float);

    // Host allocations
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_D = (float *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;           // Example values: 0, 1, 2, ...
        h_B[i] = (float)(N - i);     // Example values: N, N-1, ...
    }

    // Device allocations
    float *d_A, *d_B, *d_D;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_D, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Define scalar
    float scalar = 2.5f;

    // Kernel launch parameters
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernel
    vecAddMul<<<gridSize, blockSize>>>(d_A, d_B, d_D, scalar, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost));

    // Print first 10 results to verify
    printf("First 10 results of D[i] = (A[i] + B[i]) * %f:\n", scalar);
    for (int i = 0; i < 10; ++i) {
        printf("D[%d] = %f\n", i, h_D[i]);
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_D));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_D);

    return 0;
}
```