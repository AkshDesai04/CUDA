/*
Aim of the program:
Implement C[i] = B[i] - A[i].

Thinking:
The task is to write a simple CUDA program that performs element‑wise subtraction of two arrays A and B, storing the result in array C.  The program will:
1. Allocate host memory for A, B, and C.
2. Initialize A and B with some test values.
3. Allocate device memory for A, B, and C.
4. Copy A and B from host to device.
5. Launch a CUDA kernel where each thread computes C[i] = B[i] - A[i] for its index.
6. Copy the result array C back to host.
7. Optionally, verify the result by printing the first few elements.
8. Free all allocated memory.
The kernel will be a simple one‑dimensional grid of threads; each thread handles a single array element.  The launch configuration (threads per block and blocks per grid) will be chosen based on the array size.  Basic CUDA error checking is included to catch allocation and launch failures.  The code is self‑contained and uses only standard CUDA runtime APIs.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel to compute C[i] = B[i] - A[i]
__global__ void subtractKernel(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = B[idx] - A[idx];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                        \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

int main(void)
{
    const int N = 1024;               // Size of the arrays
    const int SIZE = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(SIZE);
    float *h_B = (float *)malloc(SIZE);
    float *h_C = (float *)malloc(SIZE);

    // Initialize input arrays
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;           // Example: A[i] = i
        h_B[i] = (float)(2 * i);     // Example: B[i] = 2*i
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_B, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_C, SIZE));

    // Copy host arrays to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    subtractKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());   // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, SIZE, cudaMemcpyDeviceToHost));

    // Verify and print first 10 results
    printf("First 10 results of C[i] = B[i] - A[i]:\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
