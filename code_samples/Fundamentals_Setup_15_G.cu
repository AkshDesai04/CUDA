```c
/*
Is `cudaDeviceReset()` necessary for a program that cleans up all its resources properly with `cudaFree`? Explain in a comment.

Thinking:

1. In CUDA, resources such as device memory, streams, events, and contexts are allocated and released by the runtime.
2. If a program explicitly frees all device memory with cudaFree and destroys all other CUDA objects, the GPU context will remain until the program exits.
3. When the host program terminates normally, the operating system will reclaim all resources, and the GPU driver will clean up the context automatically.
4. Therefore, cudaDeviceReset() is not strictly required for correct memory deallocation if the program cleans up everything before exiting.
5. However, cudaDeviceReset() has several practical benefits:
   - It forces the destruction of the CUDA context, which can be useful for profiling and debugging tools (e.g., nvprof, Nsight Systems) that rely on a clean context state.
   - It ensures that any pending asynchronous operations are completed and the context is destroyed cleanly, avoiding dangling resources that might otherwise remain in the driver.
   - It provides a clear separation point for multi-device applications or when the same process will be reused to launch multiple independent CUDA workloads.
6. In short, cudaDeviceReset() is optional for programs that already free all resources, but it is good practice to include it, especially in production code or when profiling.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addKernel(int *c, const int *a, const int *b, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);

    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify result for a few elements
    for (int i = 0; i < 5; ++i) {
        printf("c[%d] = %d (expected %d)\n", i, h_c[i], h_a[i] + h_b[i]);
    }

    // Clean up device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Optional: reset the device to destroy the context cleanly.
    cudaDeviceReset();

    // Clean up host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```