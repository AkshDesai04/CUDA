/*
Make the function take `const char* kernel_name`, `const char* file`, and `int line` as arguments so it can print a more informative error message.

Thoughts:
- The goal is to create a CUDA program that demonstrates how to pass the kernel name, file name, and line number to an error-checking function.
- We will write a simple vector addition kernel and a host routine that allocates memory, copies data, launches the kernel, and then calls our custom error-checking function.
- The error-checking function will use `cudaGetLastError()` to capture any launch errors and print a detailed message including the kernel name, file, and line number.
- To make usage convenient, a macro `CUDA_CHECK_ERROR` will be defined that automatically passes `__FILE__` and `__LINE__`.
- The program will be self-contained and compile as a .cu file. No external files are required.

The final code below fulfills these requirements.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Custom error-checking function
void checkCUDAError(const char *kernel_name, const char *file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "CUDA error after launching kernel '%s' at %s:%d: %s\n",
                kernel_name,
                file,
                line,
                cudaGetErrorString(err));
        // Optional: abort the program
        exit(EXIT_FAILURE);
    }
}

// Macro to simplify error checking with automatic file and line
#define CUDA_CHECK_ERROR(kernel_name) checkCUDAError(kernel_name, __FILE__, __LINE__)

// Simple vector addition kernel
__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CUDA_CHECK_ERROR("vecAdd");

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result
    int errors = 0;
    for (int i = 0; i < N; ++i)
    {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5)
        {
            ++errors;
            if (errors < 10)
                printf("Mismatch at %d: %f != %f\n", i, h_C[i], expected);
        }
    }

    if (errors == 0)
        printf("Success! All results match.\n");
    else
        printf("Failed with %d mismatches.\n", errors);

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
