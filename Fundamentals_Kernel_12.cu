/*
Store the grid and block dimensions in `dim3` variables on the host before launching the kernel. Launch the kernel using these variables, e.g., `myKernel<<<gridDim, blockDim>>>();`. This is the standard way to configure a launch.

Thinking:
- The goal is to demonstrate the correct way to set up grid and block dimensions using dim3 variables on the host.
- We'll implement a simple element‑wise addition kernel (`myKernel`) that adds two input arrays `a` and `b` into an output array `c`.
- On the host, we allocate host and device memory, initialize the input arrays, and copy them to the device.
- We choose a block size (e.g., 256 threads) and compute the number of blocks needed to cover all elements.
- We store the block size in a `dim3 blockDim` variable and the computed block count in a `dim3 gridDim` variable.
- The kernel launch syntax uses these variables: `myKernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);`.
- After the kernel completes, we copy the result back to the host and optionally print a few values for verification.
- Finally we free device memory and exit.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Kernel that performs element-wise addition: c[i] = a[i] + b[i] */
__global__ void myKernel(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

/* Helper macro for checking CUDA errors */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",     \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),          \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void)
{
    const int N = 1 << 20; // Number of elements (e.g., 1 million)
    const int size = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, size));
    CUDA_CHECK(cudaMalloc((void **)&d_b, size));
    CUDA_CHECK(cudaMalloc((void **)&d_c, size));

    /* Copy inputs to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    /* Define block and grid dimensions */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(blocksPerGrid);

    /* Launch the kernel using the dim3 variables */
    myKernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Wait for GPU to finish */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    /* Simple verification: print first 5 results */
    printf("First 5 results of c[i] = a[i] + b[i]:\n");
    for (int i = 0; i < 5; ++i) {
        printf("c[%d] = %f\n", i, h_c[i]);
    }

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    /* Free host memory */
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
