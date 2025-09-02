/*
Use `dim3` variables to define a grid of (1,1,1) and a block of (64,1,1) and launch the kernel.
Thinking: The program demonstrates how to set up and launch a CUDA kernel using dim3 for
grid and block dimensions. We create a simple kernel that writes the global thread index
to an array. The grid dimension is 1x1x1 and the block dimension is 64x1x1, meaning
64 threads in total. We allocate host and device memory, launch the kernel,
copy back the result, and print the values to verify the launch configuration.
The code includes basic error checking for CUDA calls.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void writeThreadIdx(int *arr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    arr[idx] = idx;
}

void checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error after %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    const int N = 64;
    int *h_arr = (int *)malloc(N * sizeof(int));
    int *d_arr = nullptr;

    // Allocate device memory
    cudaMalloc((void **)&d_arr, N * sizeof(int));
    checkCudaError("cudaMalloc");

    // Define grid and block dimensions using dim3
    dim3 blockDim(64, 1, 1);
    dim3 gridDim(1, 1, 1);

    // Launch kernel
    writeThreadIdx<<<gridDim, blockDim>>>(d_arr);
    checkCudaError("Kernel launch");

    // Copy result back to host
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy to host");

    // Print results
    printf("Thread indices written by kernel:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}
