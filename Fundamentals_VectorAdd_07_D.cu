/*
Handle `double` precision data.

In this program I demonstrate how to work with double-precision floating-point numbers in CUDA.  
The main goal is to show that device kernels can accept and correctly compute with `double`
arguments, how to transfer double arrays between host and device, and how to retrieve the
results back on the CPU.  The program performs a simple element-wise addition of two
double arrays: C[i] = A[i] + B[i].  A few important design decisions were made:

1. **Kernel implementation**  
   The kernel is written with `__global__` and uses a 1D grid of threads.  
   Each thread processes a single element, reading from `A` and `B` and writing to `C`.  
   No shared memory is needed because the operation is trivially parallel.

2. **Memory allocation and transfer**  
   Host memory is allocated with `malloc` and initialized with some values.  
   Device memory is allocated with `cudaMalloc`.  
   `cudaMemcpy` is used for copying data back and forth.  
   For large arrays, pinned memory (`cudaMallocHost`) could be used to accelerate transfers,
   but for simplicity we stick to pageable memory.

3. **Error checking**  
   A helper macro `CUDA_CHECK` calls `cudaGetLastError()` after each CUDA API call to
   ensure that any error is caught early.  This aids debugging especially when working
   with double precision, as some GPUs may not support it or may require higher compute
   capability.

4. **Compute capability**  
   Double precision support is guaranteed on devices with compute capability >= 1.3.
   The kernel is launched with a 256-thread block size; this is a common choice that
   works well on most modern GPUs.  For older GPUs you may want to reduce this number.

5. **Result verification**  
   After copying the result back to the host, the program prints the first ten elements
   of each array to confirm the addition was performed correctly.

6. **Compilation**  
   The program should be compiled with an appropriate compute capability flag, e.g.  
   `nvcc -arch=sm_20 -o double_add double_add.cu`.  
   If the target GPU does not support double precision, you will receive an error during
   compilation or execution.

The code below follows the standard structure of a CUDA C program, is self-contained,
and focuses on double-precision arithmetic.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel to add two double arrays
__global__ void vecAddDouble(const double* A, const double* B, double* C, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(double);

    // Allocate host memory
    double *h_A = (double*)malloc(bytes);
    double *h_B = (double*)malloc(bytes);
    double *h_C = (double*)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (double)i;
        h_B[i] = (double)(N - i);
    }

    // Allocate device memory
    double *d_A = NULL;
    double *d_B = NULL;
    double *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAddDouble<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("First 10 results of A + B:\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f + %f = %f\n", i, h_A[i], h_B[i], h_C[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Double precision vector addition completed successfully.\n");
    return EXIT_SUCCESS;
}
