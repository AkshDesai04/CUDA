/*
Aim of the program (as passed in the prompt word to word):
Call `cudaDeviceReset()` at the very beginning of your `main` function to ensure a clean state.

Thinking:
- The user explicitly wants to see the call to cudaDeviceReset() at the start of main.
- We need to produce a complete CUDA C (.cu) file that compiles and runs.
- A simple example is a vector addition kernel that demonstrates device reset usage.
- Include necessary headers, error checking, memory allocation, kernel launch, data transfer, and cleanup.
- The call to cudaDeviceReset() is placed as the very first statement in main, before any other CUDA API calls, to ensure the device is in a clean state.
- Provide minimal but functional code that can be compiled with nvcc.
- The code will be self-contained and should print a small result to verify execution.
- All comments and output are contained within the .cu file as requested, with no extra explanatory text.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel to add two arrays
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void)
{
    // Ensure a clean device state
    CUDA_CHECK(cudaDeviceReset());

    const int N = 10;
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float h_A[N], h_B[N], h_C[N];
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Result of vector addition:\n");
    for (int i = 0; i < N; ++i)
        printf("C[%d] = %f\n", i, h_C[i]);

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Final device reset (optional)
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
