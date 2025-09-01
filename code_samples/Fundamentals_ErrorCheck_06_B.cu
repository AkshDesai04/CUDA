```cu
/*
Make the function take `const char* kernel_name`, `const char* file`, and `int line` as arguments so it can print a more informative error message.

Thinking:
- The goal is to modify a CUDA error-checking helper so that it receives the name of the kernel that was launched, the source file, and the line number.  
- This allows error messages to reference the exact location in the host code where a kernel launch occurred, making debugging easier.  
- We will implement a helper function `checkCudaError` that accepts `cudaError_t err`, `const char* kernel_name`, `const char* file`, and `int line`.  
- Inside the function we will compare the error code with `cudaSuccess`. If it differs, we print a detailed message including the kernel name, file, line, and the human-readable error string returned by `cudaGetErrorString(err)`.  
- To make usage convenient, we will define a macro `CUDA_CHECK` that automatically passes `__FILE__` and `__LINE__`. The caller supplies the kernel name as a string literal.  
- A small example kernel will be provided (simple vector addition) to demonstrate how the macro is used after a kernel launch.  
- The program will allocate host and device memory, launch the kernel, check for errors with the new helper, copy results back, and print them.  
- All standard CUDA headers are included, and error checking is performed for memory allocations and kernel execution.  
- The code is fully self-contained and can be compiled with `nvcc` to produce an executable that demonstrates the improved error reporting.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking helper that takes kernel name, file, and line number */
void checkCudaError(cudaError_t err, const char* kernel_name, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "CUDA error in kernel '%s' at %s:%d - %s\n",
                kernel_name, file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* Macro to simplify usage of checkCudaError */
#define CUDA_CHECK(err, kernel_name) \
    checkCudaError((err), (kernel_name), __FILE__, __LINE__)

/* Simple vector addition kernel */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;          // 1M elements
    const size_t size = N * sizeof(float);

    /* Host memory allocation */
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    /* Device memory allocation */
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size), "cudaMalloc d_A");
    CUDA_CHECK(cudaMalloc((void**)&d_B, size), "cudaMalloc d_B");
    CUDA_CHECK(cudaMalloc((void**)&d_C, size), "cudaMalloc d_C");

    /* Copy host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D d_A");
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D d_B");

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError(), "vectorAdd kernel");

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H d_C");

    /* Verify a few results */
    int correct = 1;
    for (int i = 0; i < 10; ++i)
    {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f)
        {
            fprintf(stderr, "Mismatch at index %d: got %f, expected %f\n", i, h_C[i], expected);
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("First 10 results verified successfully.\n");

    /* Clean up */
    CUDA_CHECK(cudaFree(d_A), "cudaFree d_A");
    CUDA_CHECK(cudaFree(d_B), "cudaFree d_B");
    CUDA_CHECK(cudaFree(d_C), "cudaFree d_C");
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
```