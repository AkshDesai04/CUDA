/*
Implement for `double` data type.
The goal of this program is to demonstrate a simple CUDA kernel that operates on double precision floating point numbers.  
We will perform vector addition: c = a + b for two input vectors of type double.  
The program follows these steps:

1. Allocate host memory for two input vectors `a` and `b`, and an output vector `c`.
2. Initialize the input vectors with deterministic values for easy verification.
3. Allocate device memory for the vectors.
4. Copy the input vectors from host to device.
5. Launch a CUDA kernel that adds corresponding elements of `a` and `b` and stores the result in `c`.  
   The kernel uses a simple linear index scheme and checks bounds.
6. Copy the result vector `c` back to host.
7. Verify the results on the host and report success/failure.
8. Clean up all allocated memory.

The code includes basic CUDA error checking to make debugging easier.  
All operations are performed with the `double` data type, ensuring that the program uses double-precision arithmetic on the GPU.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors following a CUDA API call or kernel launch */
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* Kernel that performs element-wise addition of two double arrays */
__global__ void vectorAdd(const double *a, const double *b, double *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    /* Size of vectors */
    const int N = 1 << 20;              /* 1M elements */
    const size_t bytes = N * sizeof(double);

    /* Allocate host memory */
    double *h_a = (double *)malloc(bytes);
    double *h_b = (double *)malloc(bytes);
    double *h_c = (double *)malloc(bytes);
    if (!h_a || !h_b || !h_c)
    {
        fprintf(stderr, "Failed to allocate host vectors\n");
        exit(EXIT_FAILURE);
    }

    /* Initialize input vectors */
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = (double)i;
        h_b[i] = (double)(N - i);
    }

    /* Allocate device memory */
    double *d_a = NULL;
    double *d_b = NULL;
    double *d_c = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_b, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_c, bytes));

    /* Copy input data from host to device */
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /* Determine block and grid sizes */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    /* Check for kernel launch errors */
    CHECK_CUDA(cudaGetLastError());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    /* Verify results */
    int success = 1;
    for (int i = 0; i < N; ++i)
    {
        double expected = h_a[i] + h_b[i];
        if (h_c[i] != expected)
        {
            fprintf(stderr, "Mismatch at index %d: GPU result %f, expected %f\n",
                    i, h_c[i], expected);
            success = 0;
            break;
        }
    }

    if (success)
        printf("Vector addition successful. All results match.\n");
    else
        printf("Vector addition failed.\n");

    /* Free device memory */
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    /* Free host memory */
    free(h_a);
    free(h_b);
    free(h_c);

    /* Reset device (optional) */
    CHECK_CUDA(cudaDeviceReset());

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
