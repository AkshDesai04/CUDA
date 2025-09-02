/*
 * Aim of the program: Wrap the `cudaMalloc` calls.
 *
 * Thinking:
 * The user requested a CUDA program that demonstrates wrapping the cudaMalloc
 * calls.  In CUDA, cudaMalloc returns a cudaError_t which should be checked.
 * A common pattern is to wrap cudaMalloc in a helper function that does the
 * allocation and immediately checks for errors, possibly printing an error
 * message and exiting if the allocation fails.  This wrapper can also log the
 * allocation size or keep track of allocations for debugging.
 *
 * In this example I define a helper function `cudaMallocWrapper` that takes a
 * pointer to a device pointer and a size.  It calls cudaMalloc, checks the
 * result, and if an error occurs it prints a message and aborts.  This
 * function then returns the allocated device pointer for convenience.
 *
 * The main program demonstrates allocating an integer array of a given size,
 * initializing it on the host, copying it to the device, performing a trivial
 * kernel that multiplies each element by 2, copying the results back, and
 * printing them.  All cudaMalloc calls are made through the wrapper.
 *
 * The code is self‑contained, uses the CUDA runtime API, and can be compiled
 * with `nvcc` into a .cu file.  No external libraries are required.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Wrapper for cudaMalloc that checks for errors and prints a message */
void* cudaMallocWrapper(size_t size, const char* name)
{
    void* devPtr = NULL;
    cudaError_t err = cudaMalloc(&devPtr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error allocating %s (%zu bytes): %s\n",
                name, size, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return devPtr;
}

/* Simple kernel that multiplies each element by 2 */
__global__ void multiplyByTwo(int* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] *= 2;
}

int main(void)
{
    const int N = 10;
    int h_data[N];

    /* Initialize host data */
    for (int i = 0; i < N; ++i)
        h_data[i] = i + 1;

    /* Allocate device memory using the wrapper */
    int* d_data = (int*)cudaMallocWrapper(N * sizeof(int), "int array");

    /* Copy data from host to device */
    cudaError_t err = cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error copying data to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        exit(EXIT_FAILURE);
    }

    /* Launch kernel: 2 threads per block for simplicity */
    int threadsPerBlock = 2;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    multiplyByTwo<<<blocks, threadsPerBlock>>>(d_data, N);

    /* Check for kernel launch errors */
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        exit(EXIT_FAILURE);
    }

    /* Copy result back to host */
    err = cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error copying data to host: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        exit(EXIT_FAILURE);
    }

    /* Print the results */
    printf("Result after kernel execution:\n");
    for (int i = 0; i < N; ++i)
        printf("%d ", h_data[i]);
    printf("\n");

    /* Clean up */
    cudaFree(d_data);

    return 0;
}
