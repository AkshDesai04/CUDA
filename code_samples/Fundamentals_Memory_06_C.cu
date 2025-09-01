/*
 * Aim: Explain in a comment why memory leaks are bad, especially on a shared resource like a GPU.
 *
 * Memory leaks on a GPU are particularly harmful because GPU memory is a highly limited
 * shared resource.  Each GPU has a finite amount of global memory that is shared among
 * all processes and threads that run on that device.  If a program or kernel allocates
 * memory and fails to free it, that memory remains reserved until the program exits or
 * the CUDA context is destroyed.  In a multi‑tenant environment, such as a cloud
 * provider or a shared workstation, one process leaking memory can quickly consume
 * all available GPU memory, preventing other processes from allocating the space they
 * need.  This leads to reduced performance, increased latency, and eventually kernel
 * launch failures or application crashes.  Even in a single‑process scenario, a leak
 * can cause the device to run out of memory after a few iterations of a loop, causing
 * the application to abort unexpectedly.  Since GPUs are often used for
 * performance‑critical workloads, any unnecessary consumption of memory can degrade
 * overall system throughput.  Therefore, it is essential to explicitly free GPU
 * allocations (e.g., with cudaFree) and to use RAII‑style wrappers or smart pointers
 * whenever possible to prevent leaks.
 */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel(int *data) {
    // Simple kernel that writes thread index to the array
    data[threadIdx.x] = threadIdx.x;
}

int main() {
    int *devPtr = nullptr;
    const int N = 1024;
    size_t size = N * sizeof(int);

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&devPtr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Launch kernel
    dummyKernel<<<1, N>>>(devPtr);
    cudaDeviceSynchronize();

    // Intentionally not freeing device memory to illustrate a memory leak.
    // In a production program, you should call cudaFree(devPtr);
    // e.g., cudaFree(devPtr);

    return EXIT_SUCCESS;
}
