/*
Instead of a second host array, copy the data back into the original host array.

Thinking:
The goal is to illustrate a CUDA program that avoids allocating an additional host array to receive results from the device. 
We will allocate a single host array 'h_data', copy it to device memory 'd_data', launch a simple kernel that processes each element, 
then copy the processed data back into the same host array 'h_data'. 
This demonstrates in-place data transfer on the host side. 
The code includes error checking macros, a basic kernel that adds 1 to each element, and the standard CUDA host-device memory copy flow.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

__global__ void increment_kernel(int *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1;
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_data = (int *)malloc(size);
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }

    // Allocate device memory
    int *d_data = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_data, size));

    // Copy host data to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back into the same host array (no second host array)
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("Sample outputs after increment:\n");
    for (int i = 0; i < 5; ++i) {
        printf("h_data[%d] = %d\n", i, h_data[i]);
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);

    printf("Program completed successfully.\n");
    return EXIT_SUCCESS;
}
