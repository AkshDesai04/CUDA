/*
Check and print if the device can overlap a `memcpy` operation with kernel execution (`deviceOverlap`).

Thoughts:
- The CUDA runtime provides the attribute `cudaDevAttrDeviceOverlap` via `cudaDeviceGetAttribute`.
- This attribute returns 1 if the device can overlap memory copies with kernel execution, otherwise 0.
- The simplest program queries this attribute and prints a message indicating the capability.
- To keep the example realistic, a minimal kernel is defined and launched in a separate stream. 
  While the kernel runs, we issue an asynchronous memcpy in the default stream; however, this 
  demo is only for illustration – the real check is performed by reading the attribute.
- We include the standard CUDA runtime header and use `cudaDeviceSynchronize` to ensure 
  proper cleanup before exiting.
*/

#include <cuda_runtime.h>
#include <stdio.h>

// Simple kernel that does nothing but ensures a kernel launch
__global__ void dummyKernel(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 10) {
        data[idx] = data[idx] * 2;
    }
}

int main(void) {
    int canOverlap = 0;
    cudaError_t err = cudaDeviceGetAttribute(&canOverlap, cudaDevAttrDeviceOverlap, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (canOverlap) {
        printf("Device can overlap memcpy with kernel execution.\n");
    } else {
        printf("Device cannot overlap memcpy with kernel execution.\n");
    }

    // Allocate device memory for dummy kernel
    int *d_data;
    size_t size = 10 * sizeof(int);
    err = cudaMalloc((void**)&d_data, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Initialize host data
    int h_data[10];
    for (int i = 0; i < 10; ++i) h_data[i] = i;

    // Asynchronously copy to device
    err = cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Launch kernel in a separate stream
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    dummyKernel<<<1, 10, 0, stream>>>(d_data);

    // Synchronize to ensure all operations complete
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_data);

    // Final synchronization to catch any remaining errors
    cudaDeviceSynchronize();

    return 0;
}
