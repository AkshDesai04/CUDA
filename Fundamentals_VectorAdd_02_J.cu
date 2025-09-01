/*
Aim of the program:
Run the program with `nvprof` or `nsight compute` to profile its performance.

Thinking:
The user wants a simple CUDA C program that can be compiled with nvcc and then executed while being profiled using nvprof or Nsight Compute.  
To make the program useful for profiling, it should contain a kernel that does non-trivial work so that there is something to measure.  
A classic example is a vector addition kernel that adds two large arrays element‑wise.  
The program will:
1. Allocate host and device memory for two input vectors and one output vector.
2. Initialize the input vectors on the host.
3. Copy inputs to the device.
4. Launch the kernel with a grid/ block configuration that covers all elements.
5. Copy the result back to the host.
6. Verify the result (optional, but keeps the code self‑contained).
7. Clean up resources.

The code is kept minimal and self‑contained so it can be compiled with `nvcc` and run on any GPU that supports CUDA.  
Once compiled, the user can profile it with:
```
nvprof ./vector_add
```
or
```
nsight-sys ./vector_add
```
or the newer Nsight Compute GUI by launching the executable from there.  
The kernel is intentionally simple but still large enough (e.g., 1<<20 elements) to give measurable execution times.  
The code is written in standard CUDA C, using modern C++ constructs for convenience but still compiling with a C compiler if needed.  
*/

// vector_add.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Size of vectors (1 million elements)
#define N (1 << 20)   // 1,048,576

// CUDA kernel for vector addition: C = A + B
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

// Utility to check CUDA errors
#define CHECK_CUDA(call)                                                    \
    {                                                                        \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    }

int main(void)
{
    float *h_A = NULL;
    float *h_B = NULL;
    float *h_C = NULL;
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Optional: verify results
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            fprintf(stderr, "Mismatch at index %d: GPU %f, CPU %f\n",
                    i, h_C[i], expected);
            correct = false;
            break;
        }
    }
    if (correct)
        printf("Vector addition successful and verified.\n");
    else
        printf("Vector addition failed verification.\n");

    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset device
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
