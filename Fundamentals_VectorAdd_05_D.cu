```cuda
/*
Aim: Implement C[i] = A[i] * B[i] * D[i] (three-vector multiplication).

Thinking:
- This program demonstrates element‑wise multiplication of three vectors on the GPU.
- We write a simple CUDA kernel that multiplies corresponding elements of
  three input arrays A, B, and D and writes the result into array C.
- Each thread handles one element; the kernel guards against out‑of‑bounds
  accesses by checking the global index against the vector length N.
- The host code:
  1. Allocates and initializes host vectors with sample data.
  2. Allocates device memory for all five vectors.
  3. Copies input data from host to device.
  4. Launches the kernel with an appropriate grid/block configuration.
  5. Synchronizes, copies the result back to host, and prints a few entries
     for verification.
  6. Frees all allocated memory and finalizes the program.
- Error checking is performed after each CUDA API call to catch issues
  early and provide informative messages.
- The code is self‑contained, uses only standard CUDA headers, and can
  be compiled with `nvcc -o vector_mul vector_mul.cu`.
*/

#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                    \
        }                                                           \
    } while (0)

// Kernel: element‑wise multiplication of three vectors
__global__ void mul3(const float *A, const float *B, const float *D,
                     float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx] * D[idx];
    }
}

int main(void)
{
    const int N = 1024;                 // Vector length
    const int bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_D = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;          // Example: 0,1,2,...
        h_B[i] = (float)(i + 1);    // Example: 1,2,3,...
        h_D[i] = (float)(i + 2);    // Example: 2,3,4,...
    }

    // Allocate device memory
    float *d_A, *d_B, *d_D, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_D, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_D, h_D, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    mul3<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Print first 10 results for verification
    printf("First 10 results of C[i] = A[i] * B[i] * D[i]:\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_D);
    free(h_C);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_D));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```