/*
Aim: Calculate the size in bytes required for the allocation (256 * sizeof(int)) and use that variable in both cudaMalloc and cudaMemcpy.

Thinking:
- We want to demonstrate proper calculation of the memory size needed for an array of 256 integers.
- The size in bytes is simply 256 multiplied by sizeof(int). We'll store this in a variable called `bytes`.
- This `bytes` variable will be reused in both the device memory allocation (`cudaMalloc`) and the memory copy operations (`cudaMemcpy`) to avoid hard‑coding the value multiple times and to make the code easier to maintain.
- For completeness, a simple kernel will increment each element of the array on the device. We then copy the data back to the host and print a few elements to verify the operation.
- Error checking after each CUDA API call ensures that any problems are reported immediately.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 256          // number of integers

__global__ void increment(int *d_arr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_arr[idx] += 1;
}

int main(void)
{
    int *h_arr = NULL;
    int *d_arr = NULL;
    size_t bytes = N * sizeof(int);  // calculate size in bytes

    // Allocate and initialize host memory
    h_arr = (int*)malloc(bytes);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }
    for (int i = 0; i < N; ++i)
        h_arr[i] = i;

    // Allocate device memory using the calculated byte size
    cudaError_t err = cudaMalloc((void**)&d_arr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return 1;
    }

    // Copy data from host to device using the same byte size
    err = cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Launch kernel: 256 threads in one block
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment<<<blocksPerGrid, threadsPerBlock>>>(d_arr);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Print first 10 elements to verify
    printf("First 10 elements after increment:\n");
    for (int i = 0; i < 10; ++i)
        printf("%d ", h_arr[i]);
    printf("\n");

    // Clean up
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}
