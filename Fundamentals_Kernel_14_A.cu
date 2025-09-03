/*
Aim: Trigger `cudaErrorInvalidConfiguration` by requesting more shared memory per block than is available on the device.

Thinking: 
To produce a cudaErrorInvalidConfiguration we need to launch a kernel with a dynamic shared memory size that exceeds the maximum shared memory per block allowed by the GPU. The CUDA runtime provides the limit through cudaDeviceGetAttribute with the attribute cudaDevAttrMaxSharedMemoryPerBlock (or cudaDevAttrMaxSharedMemoryPerBlockOptin). We query this limit, then deliberately request a larger amount (e.g., limit + 1024 bytes). The kernel itself can be a trivial one that uses extern __shared__ to indicate dynamic shared memory usage. When we launch the kernel with <<<blocks, threads, requestedSharedMemory>>> where requestedSharedMemory exceeds the limit, the launch should fail with cudaErrorInvalidConfiguration. After the launch we call cudaGetLastError() to capture the error and print the error code and message. This demonstrates the error condition directly and ensures the program ends with a clear diagnostic. No other supporting information is included; the code is self-contained and can be compiled with nvcc.
*/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void dummyKernel()
{
    extern __shared__ int shared[];
    // Just touch one element to ensure shared memory is actually referenced.
    if (threadIdx.x == 0)
        shared[0] = 0;
}

int main()
{
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get current device: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return -1;
    }

    size_t maxSharedMem;
    err = cudaDeviceGetAttribute((int*)&maxSharedMem,
                                 cudaDevAttrMaxSharedMemoryPerBlock,
                                 device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get max shared memory per block: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Device %d (%s): Max shared memory per block = %zu bytes\n", device, prop.name, maxSharedMem);

    // Request more shared memory than allowed to trigger cudaErrorInvalidConfiguration.
    size_t requestedSharedMem = maxSharedMem + 1024; // 1KB over the limit

    printf("Launching kernel with %zu bytes of dynamic shared memory per block.\n", requestedSharedMem);

    dummyKernel<<<1, 1, requestedSharedMem>>>();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed with error: %s (code %d)\n", cudaGetErrorString(err), err);
    } else {
        printf("Kernel launched successfully (unexpected).\n");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel execution error: %s (code %d)\n", cudaGetErrorString(err), err);
        } else {
            printf("Kernel executed successfully.\n");
        }
    }

    return 0;
}
