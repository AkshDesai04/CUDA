/*
A kernel can optionally take shared memory size and a stream as launch parameters: `kernel<<<grid, block, shared_mem_bytes, stream>>>();`. Practice by setting the shared memory size to 0 and the stream to `0` (default stream).
This program demonstrates how to launch a CUDA kernel with explicit shared memory size and stream parameters. The goal is to show that you can specify a shared memory allocation of zero bytes and use the default stream (stream 0) for kernel execution. The kernel itself is trivial: it increments each element of an array by one. We allocate an array on the device, copy initial data from the host, launch the kernel with the specified launch configuration, copy the results back, and print the first few values to confirm that the kernel ran correctly. A simple error‑checking macro is included to keep the code clean and to help catch any issues during CUDA API calls. The shared memory size is set to 0 because this kernel does not require any dynamic shared memory. The stream parameter is explicitly set to 0 to illustrate that the default stream is used. This code can be compiled with `nvcc` and executed on any system with an appropriate CUDA driver and toolkit installed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Simple kernel that increments each element of the array
__global__ void increment_kernel(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&data[idx], 1);  // use atomic to avoid race conditions
    }
}

int main(void) {
    const int N = 1024;                     // size of the array
    int h_data[N];
    for (int i = 0; i < N; ++i) h_data[i] = i;  // initialize host array

    int *d_data = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_data, N * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch configuration
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    size_t sharedMemBytes = 0;      // no dynamic shared memory required
    cudaStream_t stream = 0;        // default stream

    // Launch the kernel with explicit shared memory size and stream
    increment_kernel<<<gridSize, blockSize, sharedMemBytes, stream>>>(d_data, N);
    CHECK_CUDA(cudaGetLastError()); // check for launch errors

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print first 10 results to verify
    printf("First 10 elements after kernel launch:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_data[%d] = %d\n", i, h_data[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_data));
    return 0;
}
