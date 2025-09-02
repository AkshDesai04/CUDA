/*
Aim of the program: Create two timers to measure two different parts of your program sequentially.

Thinking:
- Use CUDA events for GPU timing because they provide accurate timing of device operations.
- First timer will measure the time taken for host-to-device memory copy (cudaMemcpy).
- Second timer will measure the execution time of a simple kernel that processes data on the GPU.
- Events are recorded before and after the operation, then cudaEventElapsedTime gives the elapsed time in milliseconds.
- Since the operations are sequential, no need for overlapping or streams; single stream suffices.
- Use a simple kernel that increments each element of an array.
- Allocate memory on host and device, initialize host array, copy to device, run kernel, copy back, and cleanup.
- Print elapsed times for both parts.

Potential error handling:
- Use CUDA_CHECK macro to check return status and abort on failure.
- Ensure all resources are freed.

We'll write a minimal, self-contained CUDA program that demonstrates this. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err)                                        \
    do {                                                       \
        cudaError_t err__ = (err);                             \
        if (err__ != cudaSuccess) {                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err__)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

__global__ void incKernel(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1;
    }
}

int main(void) {
    const int N = 1 << 20;  // 1M elements
    size_t size = N * sizeof(int);

    // Host memory allocation and initialization
    int *h_arr = (int *)malloc(size);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_arr[i] = 0;
    }

    // Device memory allocation
    int *d_arr;
    CUDA_CHECK(cudaMalloc((void **)&d_arr, size));

    // CUDA events for timing
    cudaEvent_t startMem, stopMem;
    cudaEvent_t startKernel, stopKernel;
    CUDA_CHECK(cudaEventCreate(&startMem));
    CUDA_CHECK(cudaEventCreate(&stopMem));
    CUDA_CHECK(cudaEventCreate(&startKernel));
    CUDA_CHECK(cudaEventCreate(&stopKernel));

    // Measure host-to-device memcpy
    CUDA_CHECK(cudaEventRecord(startMem, 0));
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stopMem, 0));
    CUDA_CHECK(cudaEventSynchronize(stopMem));

    float memTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&memTime, startMem, stopMem));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    CUDA_CHECK(cudaEventRecord(startKernel, 0));
    incKernel<<<gridSize, blockSize>>>(d_arr, N);
    CUDA_CHECK(cudaEventRecord(stopKernel, 0));
    CUDA_CHECK(cudaEventSynchronize(stopKernel));

    float kernelTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelTime, startKernel, stopKernel));

    // Optional: copy results back to host
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost));

    // Print timing results
    printf("Host to Device memcpy time:   %.3f ms\n", memTime);
    printf("Kernel execution time:        %.3f ms\n", kernelTime);

    // Clean up
    free(h_arr);
    CUDA_CHECK(cudaFree(d_arr));
    CUDA_CHECK(cudaEventDestroy(startMem));
    CUDA_CHECK(cudaEventDestroy(stopMem));
    CUDA_CHECK(cudaEventDestroy(startKernel));
    CUDA_CHECK(cudaEventDestroy(stopKernel));

    return 0;
}
