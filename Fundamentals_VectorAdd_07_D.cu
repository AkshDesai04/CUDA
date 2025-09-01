```cpp
/* 
Aim of the program:
Handle `double` precision data.

Thinking:
The user requested a CUDA program that handles double‑precision data. 
To satisfy this, I will write a simple vector addition example using double precision. 
The program will:
1. Allocate host arrays for two input vectors and an output vector.
2. Allocate corresponding device arrays on the GPU.
3. Copy the input data from host to device.
4. Launch a CUDA kernel that performs element‑wise addition using double precision.
5. Copy the result back from device to host.
6. Verify the result by printing a few elements.
7. Clean up all allocated memory.

The kernel will be declared with `__global__` and will use the standard grid‑block
scheduling pattern. Because the calculation is trivial, I will use a single block
of 256 threads for simplicity, but the kernel can handle any array size by computing
its global index. I will also include basic error checking for CUDA calls to
ensure robustness. The code is written in C style but placed in a `.cu` file so that
it can be compiled with `nvcc`. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel for element-wise addition of two double arrays
__global__ void addVectors(const double *a, const double *b, double *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                 \
    do {                                                 \
        cudaError_t err = call;                          \
        if (err != cudaSuccess) {                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__,                  \
                    cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                          \
        }                                                \
    } while (0)

int main(void)
{
    const int N = 1 << 20;   // 1M elements
    const int bytes = N * sizeof(double);

    // Allocate host memory
    double *h_a = (double *)malloc(bytes);
    double *h_b = (double *)malloc(bytes);
    double *h_c = (double *)malloc(bytes);

    if (!h_a || !h_b || !h_c)
    {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = (double)i;
        h_b[i] = (double)(2 * i);
    }

    // Allocate device memory
    double *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void **)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_b, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_c, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int THREADS_PER_BLOCK = 256;
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    addVectors<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    int errors = 0;
    for (int i = 0; i < 5; ++i)
    {
        double expected = h_a[i] + h_b[i];
        if (h_c[i] != expected)
        {
            printf("Mismatch at index %d: %f + %f != %f\n", i, h_a[i], h_b[i], h_c[i]);
            errors++;
        }
    }

    if (errors == 0)
    {
        printf("Vector addition succeeded. Sample outputs:\n");
        for (int i = 0; i < 5; ++i)
        {
            printf("h_c[%d] = %f\n", i, h_c[i]);
        }
    }
    else
    {
        printf("There were %d errors in vector addition.\n", errors);
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return EXIT_SUCCESS;
}
```