```cuda
/*
Aim of the program which is passed in this prompt word to word:
Make the vector size `N` and the block size `BLOCK_SIZE` configurable (e.g., using `const int` or `#define`). Calculate the grid size dynamically.

Thinking:
1. The program will perform a simple vector addition: C = A + B.
2. Both vector size (N) and CUDA block size (BLOCK_SIZE) will be defined as constants so they can be easily changed.
3. Grid size (number of blocks) will be computed at runtime based on N and BLOCK_SIZE:
   gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE.
4. Host memory will be allocated for vectors A, B, and C. Device memory will be allocated correspondingly.
5. Vectors A and B will be initialized on the host, copied to the device, the kernel will be launched, and the result copied back to the host.
6. Basic error checking will be included for CUDA API calls.
7. The program will print the first few elements of the result to verify correctness.
8. All code will be contained in a single .cu file, with the required comment block at the top.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int N = 1 << 20;          // Vector size (modifiable)
const int BLOCK_SIZE = 256;     // CUDA block size (modifiable)

__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));
    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = sinf(i);
        h_B[i] = cosf(i);
    }

    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc((void**)&d_A, N * sizeof(float)), "cudaMalloc d_A");
    checkCudaError(cudaMalloc((void**)&d_B, N * sizeof(float)), "cudaMalloc d_B");
    checkCudaError(cudaMalloc((void**)&d_C, N * sizeof(float)), "cudaMalloc d_C");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkCudaError(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy B");

    // Compute grid size dynamically
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel
    vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy C");

    // Verify result (print first 10 elements)
    printf("First 10 elements of result vector C:\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("C[%d] = %f\n", i, h_C[i]);
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
```