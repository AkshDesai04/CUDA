/*
Pass `N` as an argument to the kernel.
The aim of this program is to demonstrate how to pass a runtime parameter, `N`, to a CUDA kernel.
We allocate an array of size `N` on the host, copy it to device memory, launch a kernel that receives `N` as an argument, and perform a simple operation (doubling each element) inside the kernel.
The kernel uses the passed `N` to guard against out‑of‑bounds accesses.
After kernel execution, we copy the data back to the host and print a few results.
The program also performs basic error checking for CUDA API calls and accepts `N` as a command‑line argument.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

__global__ void kernel(int *d_arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] *= 2;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        fprintf(stderr, "N must be positive.\n");
        return EXIT_FAILURE;
    }

    // Allocate and initialize host array
    int *h_arr = (int*)malloc(N * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    // Allocate device memory
    int *d_arr = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, N * sizeof(int)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Launch kernel, passing N as an argument
    kernel<<<numBlocks, blockSize>>>(d_arr, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print first few results for verification
    int toPrint = N < 10 ? N : 10;
    printf("First %d elements after kernel execution:\n", toPrint);
    for (int i = 0; i < toPrint; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Clean up
    free(h_arr);
    CUDA_CHECK(cudaFree(d_arr));

    return EXIT_SUCCESS;
}
