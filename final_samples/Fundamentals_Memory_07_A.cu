/*
Aim of the program:
Explain in a comment the concept of separate host and device memory address spaces.

Thinking behind this comment and program:
The goal is to demonstrate that CUDA programs run with two distinct memory spaces:
1. Host (CPU) memory – the ordinary RAM that the host process uses.
2. Device (GPU) memory – a separate address space owned by the GPU.

These spaces are physically and logically separate; a pointer that refers to
memory on the host cannot be dereferenced on the device and vice versa.
To move data between them we must explicitly allocate memory in each space,
copy data using cudaMemcpy (or similar APIs), and then launch kernels that
operate on device memory.

In this small example we:
- Allocate an array on the host.
- Allocate a corresponding array on the device.
- Copy the host array to the device.
- Launch a kernel that adds 1 to each element.
- Copy the result back to the host.
- Print addresses and values to illustrate the separation of spaces.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that increments each element by 1
__global__ void increment(int *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_arr[idx] += 1;
}

int main()
{
    const int N = 10;
    const size_t size = N * sizeof(int);

    // Host memory allocation
    int *h_arr = (int*)malloc(size);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i)
        h_arr[i] = i;

    // Device memory allocation
    int *d_arr = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_arr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return 1;
    }

    // Copy from host to device
    err = cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment<<<blocks, threadsPerBlock>>>(d_arr, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Print addresses to show they are distinct
    printf("Host array pointer: %p\n", (void*)h_arr);
    printf("Device array pointer: %p (note: cannot be dereferenced by host)\n", (void*)d_arr);

    // Print results
    printf("Resulting array after kernel:\n");
    for (int i = 0; i < N; ++i)
        printf("%d ", h_arr[i]);
    printf("\n");

    // Clean up
    cudaFree(d_arr);
    free(h_arr);
    return 0;
}
