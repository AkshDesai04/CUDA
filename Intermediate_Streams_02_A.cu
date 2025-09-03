```cuda
/*
Create two streams: `cudaStream_t stream1, stream2; cudaStreamCreate(...)`.

Thought process:
1. The user wants a CUDA program that demonstrates the creation of two streams.
2. We'll need to include the CUDA runtime header, define a main function, and use
   cudaStreamCreate to instantiate two streams.
3. To make the program meaningful, we will perform asynchronous memory copies
   and kernel launches on each stream so the streams are actually used.
4. A simple kernel that increments each element of an integer array will be
   launched twice: once on stream1 and once on stream2.
5. After kernel execution we will copy the results back to the host, print
   them, and finally destroy the streams.
6. We will also perform basic error checking after each CUDA API call
   to ensure the program behaves correctly.
7. The code will be self‑contained, compile with nvcc, and produce a .cu
   file as requested.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 256

__global__ void addOne(int *d_data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_data[idx] += 1;
}

int main(void)
{
    cudaError_t err;
    int *h_a, *h_b;
    int *d_a, *d_b;
    cudaStream_t stream1, stream2;

    // Allocate host memory
    h_a = (int*)malloc(N * sizeof(int));
    h_b = (int*)malloc(N * sizeof(int));
    if (!h_a || !h_b) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = N - i;
    }

    // Allocate device memory
    err = cudaMalloc((void**)&d_a, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_a failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc((void**)&d_b, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_b failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        return -1;
    }

    // Create two streams
    err = cudaStreamCreate(&stream1);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreate stream1 failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        return -1;
    }
    err = cudaStreamCreate(&stream2);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreate stream2 failed: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream1);
        cudaFree(d_a);
        cudaFree(d_b);
        return -1;
    }

    // Asynchronous copy from host to device using stream1
    err = cudaMemcpyAsync(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync to d_a failed: %s\n", cudaGetErrorString(err));
    }

    // Asynchronous copy from host to device using stream2
    err = cudaMemcpyAsync(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice, stream2);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync to d_b failed: %s\n", cudaGetErrorString(err));
    }

    // Launch kernel on stream1
    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addOne<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_a, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch on stream1 failed: %s\n", cudaGetErrorString(err));
    }

    // Launch kernel on stream2
    addOne<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_b, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch on stream2 failed: %s\n", cudaGetErrorString(err));
    }

    // Asynchronous copy back to host using stream1
    err = cudaMemcpyAsync(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync to h_a failed: %s\n", cudaGetErrorString(err));
    }

    // Asynchronous copy back to host using stream2
    err = cudaMemcpyAsync(h_b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost, stream2);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync to h_b failed: %s\n", cudaGetErrorString(err));
    }

    // Wait for streams to finish
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Print first 10 elements of each result array
    printf("First 10 elements of result array a (stream1):\n");
    for (int i = 0; i < 10; ++i)
        printf("%d ", h_a[i]);
    printf("\n");

    printf("First 10 elements of result array b (stream2):\n");
    for (int i = 0; i < 10; ++i)
        printf("%d ", h_b[i]);
    printf("\n");

    // Clean up
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);

    return 0;
}
```