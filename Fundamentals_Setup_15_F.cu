/*
Aim of the program: Call `cudaDeviceSynchronize()` before `cudaDeviceReset()` and explain why this can be good practice.

My thinking: 
- `cudaDeviceReset()` cleans up all resources associated with the current device, including memory allocations, contexts, and any pending kernel executions. 
- If a kernel is still running or there are pending asynchronous operations, calling `cudaDeviceReset()` immediately may terminate those operations prematurely, leading to undefined behavior or crashes. 
- Therefore, calling `cudaDeviceSynchronize()` first forces the host to wait until all previously issued kernels and memory copies have finished, ensuring a clean shutdown of the device. 
- In practice this prevents race conditions, ensures that profiling tools see a complete execution trace, and allows for proper cleanup of device-side resources.
*/

#include <cuda_runtime.h>
#include <stdio.h>

// Simple kernel that writes the thread index into an array
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = a[i] + b[i];
}

int main(void)
{
    const int N = 256;
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b);

    // Synchronize: ensure kernel has finished before resetting the device
    // This is good practice because cudaDeviceReset will destroy the context and all
    // allocations. If there are pending operations, they may be aborted or result in
    // undefined behavior. By synchronizing, we guarantee that all work is done.
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(syncErr));
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            success = false;
            printf("Mismatch at %d: %d != %d\n", i, h_c[i], h_a[i] + h_b[i]);
            break;
        }
    }
    printf("%s\n", success ? "Success!" : "Failure!");

    // Reset device after synchronization
    cudaDeviceReset();

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
