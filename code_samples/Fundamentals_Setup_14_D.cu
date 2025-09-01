/*
Add a comment explaining what a CUDA stream is and how it relates to overlapping operations.

This program demonstrates the concept of CUDA streams by performing a simple data transfer and kernel launch
using a user-created stream. A CUDA stream is an ordered list of operations that the device executes in the
order they are issued. By default, operations are placed in the stream 0 (the default stream) and execute
sequentially. However, user-defined streams (stream 1, stream 2, etc.) can run concurrently with each other
and with operations in the default stream, enabling overlapping of data transfers, kernel execution, and
memory operations. When multiple streams are used, the GPU scheduler can interleave operations from
different streams, potentially improving throughput by hiding latency. In this example, we create a single
stream, perform a device memory copy, launch a trivial kernel, and copy the result back, all within that stream.
The comments in the code explain how these operations are ordered and how overlapping would occur if more
streams and operations were used.

This file contains only the necessary code to compile and run a minimal CUDA example demonstrating streams
and the accompanying explanation comment. No external files or support code are provided.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel(int *d_out)
{
    // Simple kernel that writes a value to device memory.
    // Each thread could perform more complex work in a real scenario.
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *d_out = 42;
    }
}

int main(void)
{
    const int size = sizeof(int);
    int *h_data = (int *)malloc(size);
    int *d_data;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void **)&d_data, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Create a CUDA stream (stream 1)
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Asynchronously copy data from host to device in the created stream
    err = cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpyAsync (H2D) failed: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        return 1;
    }

    // Launch the dummy kernel in the same stream
    dummyKernel<<<1, 1, 0, stream>>>(d_data);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        return 1;
    }

    // Asynchronously copy the result back to host
    err = cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpyAsync (D2H) failed: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        return 1;
    }

    // Wait for all operations in the stream to complete
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    // Check result
    printf("Result from device: %d\n", *h_data);

    // Clean up
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    free(h_data);

    return 0;
}
