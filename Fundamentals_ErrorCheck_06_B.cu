/*
Make the function take `const char* kernel_name`, `const char* file`, and `int line` as arguments so it can print a more informative error message.

Thinking:
The goal is to provide a helper that reports CUDA errors in a context-aware manner.  
By accepting the kernel name, the source file and line number, the helper can produce a log that shows exactly which kernel call failed, where in the host code the failure was detected, and the CUDA error string.  
To keep the implementation simple and portable, we:
- include <cuda_runtime.h> and <stdio.h>
- write a function `checkCudaError` that calls `cudaGetLastError()`, checks if the error is non-zero, and prints a formatted message containing the kernel name, file, and line.
- provide a macro `CUDA_CHECK(err, kernel)` that calls the helper with the stringified kernel name and automatic file/line macros.
- demonstrate usage by launching a trivial kernel and intentionally causing an error (e.g., by launching with an invalid grid size) to trigger the error handler.
The program is a complete, self‑contained .cu file that can be compiled with `nvcc` and executed on a CUDA capable device.
*/

#include <cuda_runtime.h>
#include <stdio.h>

// Helper function that prints a detailed error message
void checkCudaError(const char* kernel_name, const char* file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "CUDA error in kernel \"%s\" at %s:%d: %s\n",
                kernel_name,
                file,
                line,
                cudaGetErrorString(err));
        // Optionally, we can also abort the program
        // exit(EXIT_FAILURE);
    }
}

// Macro to wrap a kernel launch and invoke the error checker
#define CUDA_CHECK(kernel) \
    do { \
        kernel; \
        checkCudaError(#kernel, __FILE__, __LINE__); \
    } while (0)

// A simple CUDA kernel that adds two arrays
__global__ void addKernel(const float* a, const float* b, float* c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Correct launch
    CUDA_CHECK(addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N));

    // Intentionally launch with an invalid configuration to trigger an error
    // For example, use a negative block size
    CUDA_CHECK(addKernel<<<blocksPerGrid, -1>>>(d_a, d_b, d_c, N));

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify a few results
    printf("c[0] = %f, c[1] = %f, c[2] = %f\n", h_c[0], h_c[1], h_c[2]);

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
