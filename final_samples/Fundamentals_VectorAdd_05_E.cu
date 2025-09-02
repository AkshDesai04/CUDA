/*
Implement in-place multiplication: `A[i] *= B[i]`.

Thinking:
- The task is to write a CUDA program that multiplies two arrays element-wise
  and stores the result back into the first array (in-place).  
- We need a kernel that takes pointers to arrays A and B and an integer N for
  the number of elements. Each thread will handle one element if its index
  is within bounds.
- On the host side we will allocate two arrays, initialize them with some
  sample data, copy them to device memory, launch the kernel, copy the result
  back, and then print a few elements to verify correctness.
- We'll include basic error checking after each CUDA API call.
- The program is self‑contained, uses only standard C/C++ headers and the
  CUDA runtime API, and compiles to a .cu file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = (call);                                \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",    \
                    #call, __FILE__, __LINE__,                  \
                    cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

/* Kernel performing in-place multiplication */
__global__ void multiply_in_place(float *A, const float *B, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        A[idx] *= B[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;   // 1,048,576 elements
    const size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    if (!h_A || !h_B)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays with sample data */
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = 1.0f + (float)i / N;   // e.g., values from 1.0 to 2.0
        h_B[i] = 0.5f + (float)i / (2 * N); // values from 0.5 to 1.0
    }

    /* Allocate device memory */
    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Determine block and grid sizes */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    multiply_in_place<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    printf("First 10 results of in-place multiplication:\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("h_A[%d] = %f\n", i, h_A[i]);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    free(h_A);
    free(h_B);

    return 0;
}
