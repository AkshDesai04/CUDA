/*
Write a program that tries to access a device pointer from host code (e.g., `int* d_ptr; cudaMalloc(&d_ptr, ...); printf("%d", d_ptr[0]);`). Observe the segmentation fault or error this causes and explain why it happens.

Thinking:
The goal is to illustrate that device memory allocated with cudaMalloc resides in the GPU address space and is not directly accessible from the host CPU. The pointer returned by cudaMalloc is a GPU virtual address that the host CPU does not understand. When the host code attempts to dereference this pointer (e.g., `d_ptr[0]`), it accesses an invalid address in the host process memory, leading to a segmentation fault or undefined behaviour.

To demonstrate this, we will:
1. Allocate memory on the device with cudaMalloc.
2. Launch a simple kernel to initialise the memory.
3. Attempt to read the first element directly from host using `printf("%d", d_ptr[0]);`, which should crash.
4. Show the correct way to read back data: use cudaMemcpy to copy from device to host buffer before printing.

The program will include basic CUDA error checking to illustrate typical error handling patterns, but the segmentation fault will occur before any CUDA API error can be reported.

*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void initKernel(int *d_ptr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        d_ptr[idx] = idx * 10;  // arbitrary initialization
    }
}

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    const int N = 5;
    int *d_ptr = NULL;

    // Allocate device memory
    checkCudaError(cudaMalloc((void **)&d_ptr, N * sizeof(int)), "cudaMalloc");

    // Launch kernel to initialise memory
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    initKernel<<<blocks, threadsPerBlock>>>(d_ptr, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");

    // ------------------------------------------------------------------
    // Attempt to access device memory directly from host (this will crash)
    // ------------------------------------------------------------------
    printf("Attempting to read device memory directly from host (this should crash)...\n");
    printf("d_ptr[0] = %d\n", d_ptr[0]);  // <-- This line causes a segmentation fault

    // The following code will never execute due to the crash above.
    // ------------------------------------------------------------------
    // Correct way: copy data back to host and then print
    // ------------------------------------------------------------------
    int h_data[N];
    checkCudaError(cudaMemcpy(h_data, d_ptr, N * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");

    printf("Data copied back to host:\n");
    for (int i = 0; i < N; ++i)
        printf("h_data[%d] = %d\n", i, h_data[i]);

    // Clean up
    checkCudaError(cudaFree(d_ptr), "cudaFree");

    return 0;
}
