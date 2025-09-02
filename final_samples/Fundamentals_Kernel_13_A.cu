/*
Generalize the kernel to work on arrays. It should take `int* a`, `int* b`, and `int N`. Each thread `i` should compute `b[i] = a[i] + 10;`.

To fulfill this, I will implement a CUDA kernel that receives two device pointers and the size of the arrays. The kernel will calculate the global thread index, guard against out-of-bounds, and perform the addition. On the host side, I will allocate host arrays, initialize them, copy to device, launch the kernel with enough threads/blocks to cover all elements, copy back results, and finally print them to verify correctness. Standard CUDA error checks will be added for robustness. The program will be self‑contained and compile with nvcc.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* CUDA error checking macro */
#define CUDA_CHECK(err)                                         \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error: %s (err_num=%d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                     \
    }

/* Kernel: each thread adds 10 to its element */
__global__ void addTenKernel(const int *a, int *b, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        b[i] = a[i] + 10;
    }
}

int main(void)
{
    const int N = 100;            // Number of elements
    const size_t bytes = N * sizeof(int);

    /* Allocate host memory */
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    if (h_a == NULL || h_b == NULL)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Initialize host input array */
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = i;          // Example: a[i] = i
    }

    /* Allocate device memory */
    int *d_a = NULL;
    int *d_b = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));

    /* Copy input array from host to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addTenKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Wait for GPU to finish */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

    /* Print some results to verify */
    printf("First 10 results:\n");
    for (int i = 0; i < 10 && i < N; ++i)
    {
        printf("b[%d] = %d\n", i, h_b[i]);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_a);
    free(h_b);

    return EXIT_SUCCESS;
}
