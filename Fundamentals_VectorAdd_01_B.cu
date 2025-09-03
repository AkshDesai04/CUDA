```cuda
/*
Aim of the program: 
Initialize vector A with the values i and vector B with 100-i. Verify that C[i] is always 100.

Thought Process:
1. Determine a suitable vector size (N). For demonstration purposes, we use 1024.
2. Allocate host arrays A, B, and C with size N.
3. Initialize A[i] = i and B[i] = 100 - i on the host.
4. Allocate device memory for A, B, and C.
5. Copy host arrays A and B to the device.
6. Launch a simple CUDA kernel that performs element-wise addition:
      C[i] = A[i] + B[i]
7. Copy the resulting C array back to the host.
8. Verify on the host that each element of C equals 100. If any element differs,
   print an error message; otherwise, confirm success.
9. Clean up device and host memory.

The program uses a basic error checking macro to catch CUDA API errors early.
It is fully self-contained and can be compiled with `nvcc vector_addition.cu -o vector_addition`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

// CUDA kernel to add two vectors
__global__ void vectorAdd(const int *A, const int *B, int *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;
    size_t bytes = N * sizeof(int);

    // Allocate host memory
    h_A = (int *)malloc(bytes);
    h_B = (int *)malloc(bytes);
    h_C = (int *)malloc(bytes);
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = 100 - i;
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    // Copy host memory to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    int success = 1;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != 100) {
            fprintf(stderr, "Verification failed at index %d: C[%d] = %d, expected 100.\n",
                    i, i, h_C[i]);
            success = 0;
            break;
        }
    }

    if (success) {
        printf("All elements verified: C[i] == 100 for all i.\n");
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```