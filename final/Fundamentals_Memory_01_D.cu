/*
Aim: Instead of `malloc`, use `new int[256]` for the host allocation.

Thinking:
I will write a small CUDA program that demonstrates how to use `new` to allocate an array on the host, copy it to the device, run a kernel that increments each element, copy the result back, and finally clean up. The host array will be a simple `int` array of size 256. The kernel will add 1 to each element. I will use error checking for CUDA API calls, and I will free the host memory with `delete[]`. This program is intentionally simple to focus on the allocation change requested.
*/

#include <cstdio>
#include <cstdlib>

// Simple kernel that adds 1 to each element
__global__ void incrementKernel(int* d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1;
    }
}

int main()
{
    const int N = 256;
    const int bytes = N * sizeof(int);

    // Allocate host memory using new
    int* h_arr = new int[N];
    if (!h_arr) {
        fprintf(stderr, "Host allocation failed!\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    // Allocate device memory
    int* d_arr = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_arr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Device allocation failed: %s\n", cudaGetErrorString(err));
        delete[] h_arr;
        return EXIT_FAILURE;
    }

    // Copy host to device
    err = cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Memcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        delete[] h_arr;
        return EXIT_FAILURE;
    }

    // Launch kernel: 64 threads per block, enough blocks for N elements
    const int threadsPerBlock = 64;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    incrementKernel<<<blocks, threadsPerBlock>>>(d_arr, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        delete[] h_arr;
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Device synchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        delete[] h_arr;
        return EXIT_FAILURE;
    }

    // Copy device back to host
    err = cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Memcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        delete[] h_arr;
        return EXIT_FAILURE;
    }

    // Print first 10 elements to verify
    printf("First 10 elements after increment:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_arr);
    delete[] h_arr;

    return EXIT_SUCCESS;
}
