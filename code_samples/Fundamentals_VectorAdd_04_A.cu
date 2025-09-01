```cuda
/*
Aim of the program:
Implement `C[i] = B[i] - A[i]`.

My thinking:
The requirement is to perform an element‑wise subtraction of two arrays
A and B and store the result in C. CUDA is ideal for this simple
parallel reduction because each element can be computed independently.
I will write a small program that:

1. Generates two sample arrays (A and B) on the host.
2. Allocates corresponding memory on the device.
3. Copies the host arrays to the device.
4. Launches a CUDA kernel where each thread computes one element
   of C as `C[i] = B[i] - A[i]`. The kernel will include a bounds
   check so that the code works for any array length.
5. Copies the result back to the host.
6. Prints the first few elements of the result for verification.
7. Frees all allocated memory.

Error handling is wrapped in a macro `CUDA_CHECK` that aborts on
failure, simplifying the code. The kernel uses a simple
1‑D grid/block layout and a block size of 256 threads, which is
generally a good starting point. The code is written in plain C
with CUDA extensions, so it can be compiled with `nvcc -o subtract
subtract.cu`. No external dependencies are required beyond the CUDA
runtime library. The program is self‑contained and demonstrates the
requested operation in a clear, efficient manner.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA API errors */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* Kernel: compute C[i] = B[i] - A[i] */
__global__ void subtractKernel(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = B[idx] - A[idx];
    }
}

int main(void)
{
    /* Example array size */
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;          // Example: 0, 1, 2, ...
        h_B[i] = (float)(N - i);    // Example: N, N-1, N-2, ...
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Kernel launch configuration */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch the kernel */
    subtractKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());   // Check for launch errors

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verify the result (print first 10 elements) */
    printf("First 10 results of C[i] = B[i] - A[i]:\n");
    for (int i = 0; i < 10 && i < N; ++i)
    {
        printf("C[%d] = %.2f\n", i, h_C[i]);
    }

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```