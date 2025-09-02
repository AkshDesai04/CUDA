/*
Define a `__constant__` memory variable in your CUDA file (e.g., `__constant__ int my_const;`). In the host code, use `cudaMemcpyToSymbol` to copy a value (e.g., 42) into this variable. Write a kernel that reads from `my_const` and writes its value into an output array.

Thinking:
1. Declare a constant memory variable `my_const` of type `int` in global device memory.
2. In the host, allocate device memory for an output array.
3. Copy a host integer value (42) into the constant variable using `cudaMemcpyToSymbol`.
4. Write a kernel that, for each thread, writes `my_const` into the output array at the thread’s index.
5. Launch the kernel with enough blocks/threads to cover the array.
6. Copy the result back to the host and print a few entries to verify that all elements equal 42.
7. Include minimal error checking to keep the example concise.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__constant__ int my_const;          // Constant memory variable

__global__ void fill_from_const(int *out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = my_const;       // Read from constant memory
    }
}

int main(void)
{
    const int N = 256;              // Size of output array

    // Allocate host memory
    int *h_out = (int*)malloc(N * sizeof(int));
    if (!h_out) { fprintf(stderr, "Failed to allocate host memory\n"); return -1; }

    // Allocate device memory
    int *d_out;
    cudaError_t err = cudaMalloc((void**)&d_out, N * sizeof(int));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err)); return -1; }

    // Copy value 42 into constant memory
    int const_val = 42;
    err = cudaMemcpyToSymbol(my_const, &const_val, sizeof(int));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err)); return -1; }

    // Launch kernel
    int threadsPerBlock = 128;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    fill_from_const<<<blocks, threadsPerBlock>>>(d_out, N);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err)); return -1; }

    // Copy result back to host
    err = cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err)); return -1; }

    // Print first 10 elements to verify
    printf("First 10 elements of output array:\n");
    for (int i = 0; i < 10; ++i)
        printf("%d ", h_out[i]);
    printf("\n");

    // Clean up
    cudaFree(d_out);
    free(h_out);

    return 0;
}
