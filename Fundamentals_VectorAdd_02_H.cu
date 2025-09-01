/*
Aim of the program: Pass `N` as an argument to the kernel.

Thinking:
- The program will accept an integer N from the command line.
- It will allocate an array of size N on the host, initialize it with values 0..N-1.
- Allocate device memory of same size.
- Copy data from host to device.
- Launch a CUDA kernel that receives N as a parameter along with the device array.
- The kernel will process each element (e.g., multiply by 2) using the passed N to bound the loop.
- Copy results back to host and verify correctness.
- Proper error checking and memory cleanup will be performed.
- The kernel launch configuration will use a reasonable block size (e.g., 256 threads per block) and compute the number of blocks needed to cover N elements.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void processArray(int *d_arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // Example operation: double the value
        d_arr[idx] *= 2;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);
    if (N <= 0)
    {
        fprintf(stderr, "N must be a positive integer.\n");
        return EXIT_FAILURE;
    }

    size_t bytes = N * sizeof(int);

    // Allocate host memory
    int *h_arr = (int *)malloc(bytes);
    if (!h_arr)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i)
        h_arr[i] = i;

    // Allocate device memory
    int *d_arr;
    cudaError_t err = cudaMalloc((void **)&d_arr, bytes);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice);
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

    // Launch kernel, passing N as an argument
    processArray<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Copy result back to host
    err = cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Verify results
    int errors = 0;
    for (int i = 0; i < N; ++i)
    {
        if (h_arr[i] != i * 2)
        {
            if (errors < 10)
            {
                fprintf(stderr, "Mismatch at index %d: expected %d, got %d\n",
                        i, i * 2, h_arr[i]);
            }
            ++errors;
        }
    }

    if (errors == 0)
    {
        printf("All %d elements processed correctly.\n", N);
    }
    else
    {
        printf("Total mismatches: %d out of %d elements.\n", errors, N);
    }

    // Cleanup
    cudaFree(d_arr);
    free(h_arr);

    return EXIT_SUCCESS;
}
