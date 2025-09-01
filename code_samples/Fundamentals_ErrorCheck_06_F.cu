/*
In a program with multiple, sequential kernel launches, place the check after each one to pinpoint which one is failing.

Below is a complete CUDA C program (.cu file) that demonstrates this approach.  The program defines two simple kernels:

1. kernelGood – performs a benign operation (adds two arrays).
2. kernelBad  – intentionally dereferences an invalid device pointer to trigger an error.

After each kernel launch, the program calls a helper function `checkCudaError` that performs `cudaGetLastError()` and `cudaDeviceSynchronize()`.  If either of these calls indicates a failure, the function prints the name of the kernel that caused the problem and aborts execution.  This pattern allows the developer to identify exactly which kernel launch is problematic in a chain of sequential launches.

The program also shows basic host–device memory allocation, data transfer, and cleanup.  No external libraries are required – just the CUDA Runtime API.  Compile with:

    nvcc -o kernel_check kernel_check.cu

Run the executable to see the error reporting in action.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Helper function to check the result of a kernel launch
void checkCudaError(const char* kernelName) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error (%s): %s\n", kernelName,
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Synchronize to catch errors that occur during kernel execution
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution error (%s): %s\n", kernelName,
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Kernel '%s' executed successfully.\n", kernelName);
}

// A simple kernel that adds two arrays
__global__ void kernelGood(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// A kernel that intentionally causes an error by dereferencing a null pointer
__global__ void kernelBad(float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = c[idx] + 1.0f; // c is NULL -> illegal memory access
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 1.0f;
        h_b[i] = (N - i) * 1.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA(cudaMalloc((void**)&d_c, size));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernelGood
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernelGood<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    checkCudaError("kernelGood");

    // Copy result back to host and verify (optional)
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    // (Verification code omitted for brevity)

    // Intentionally trigger an error in kernelBad
    // We pass a NULL pointer for d_c to cause an illegal memory access
    kernelBad<<<blocksPerGrid, threadsPerBlock>>>(NULL, N);
    checkCudaError("kernelBad");

    // Cleanup
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    printf("Program completed successfully.\n");
    return EXIT_SUCCESS;
}
