```cpp
/* 
Aim of the program (exactly as provided in the prompt):
Handle `double` precision data.

Thinking:
The task is to produce a CUDA C program that demonstrates handling double precision data on the GPU.
The simplest yet illustrative example is a vector addition kernel where two input vectors of type double are added element-wise to produce an output vector.
This example will cover the following key aspects:

1. Host memory allocation for double arrays and initialization.
2. Device memory allocation and data transfer using cudaMalloc and cudaMemcpy.
3. Definition of a CUDA kernel that operates on double precision elements.
4. Kernel launch configuration (grid and block dimensions).
5. Transfer of the result back to host memory and validation of correctness.
6. Basic error checking for all CUDA API calls.
7. Use of CUDA event timing to measure kernel execution time (optional but useful for demonstration).

We will keep the code straightforward, focusing on correctness and clarity rather than performance tuning. The code compiles with `nvcc -arch=sm_20` (or higher) to ensure double precision support. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                  \
    {                                                                     \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                 \
    }

/* Kernel for element-wise addition of two double vectors */
__global__ void vecAddDouble(const double *A, const double *B, double *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main(void)
{
    /* Problem size */
    const int N = 1 << 20;        // 1M elements
    const int bytes = N * sizeof(double);

    /* Allocate host memory */
    double *h_A = (double*)malloc(bytes);
    double *h_B = (double*)malloc(bytes);
    double *h_C = (double*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        exit(EXIT_FAILURE);
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (double)i;
        h_B[i] = (double)(N - i);
    }

    /* Allocate device memory */
    double *d_A = NULL;
    double *d_B = NULL;
    double *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Kernel launch configuration */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Record start event */
    CUDA_CHECK(cudaEventRecord(start, 0));

    /* Launch kernel */
    vecAddDouble<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    /* Record stop event */
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    /* Compute elapsed time */
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verify result */
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        double expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-9) {
            if (errors < 10) { // print first few errors
                printf("Mismatch at index %d: GPU %f != Expected %f\n", i, h_C[i], expected);
            }
            ++errors;
        }
    }
    if (errors == 0) {
        printf("Result verification: SUCCESS\n");
    } else {
        printf("Result verification: FAILED with %d errors\n", errors);
    }

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```