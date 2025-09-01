/*
Aim: Inside the kernel, add a `printf` from a single thread (e.g., `if (global_id == 12345)`) to inspect a value.

Thinking:
The requirement is to provide a complete CUDA program in a single .cu file.  
The program should launch a kernel that performs a simple operation on an array
and includes a `printf` executed by only one thread (identified by a global thread id, e.g. 12345).  
To make the example self‑contained:

1. Allocate a small array on the host and copy it to the device.
2. Launch a 1‑D grid of blocks such that the total number of threads is at least 12346
   so that thread 12345 exists.
3. In the kernel, compute a value (e.g., multiply each element by 2) and for the
   thread with global id 12345 print the original value and the computed value.
4. Copy the result back to the host and optionally print a confirmation message.
5. Use `printf` inside the kernel; this is supported on devices of compute capability
   2.x and higher.

The code below follows this plan and is fully compilable with `nvcc`. */

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that doubles each element and prints from thread 12345
__global__ void double_and_print(int *d_arr, int N)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < N)
    {
        int original = d_arr[global_id];
        int doubled  = original * 2;
        d_arr[global_id] = doubled;

        // Print from a single thread to inspect values
        if (global_id == 12345 && global_id < N)
        {
            printf("Thread %d: original=%d, doubled=%d\n", global_id, original, doubled);
        }
    }
}

int main()
{
    const int N = 20000;                // Number of elements (>= 12346)
    const size_t size = N * sizeof(int);

    int *h_arr = (int*)malloc(size);
    if (!h_arr)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i)
        h_arr[i] = i;

    // Allocate device memory
    int *d_arr = NULL;
    cudaError_t err = cudaMalloc((void**)&d_arr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Copy data to device
    err = cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    double_and_print<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Copy result back to host
    err = cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Optional: verify one element
    if (12345 < N)
        printf("Host verification: h_arr[12345] = %d\n", h_arr[12345]);

    // Clean up
    cudaFree(d_arr);
    free(h_arr);

    return EXIT_SUCCESS;
}
