/*
Explain in a comment that because kernel launches are asynchronous, the CPU can continue working on other tasks while the GPU is busy.

Thinking: The goal of this program is to illustrate the asynchrony between CPU and GPU. In CUDA, launching a kernel is non‑blocking; it returns control to the host before the GPU finishes execution. Therefore, the host can perform other operations while the GPU is still processing the kernel. We will launch a simple kernel that writes to an array. Immediately after launching, the CPU will perform a time‑consuming loop that prints messages. We will use cudaEvent timing to demonstrate that the GPU work overlaps with CPU work. Finally, we will synchronize to ensure GPU finishes before we copy data back and exit.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// Simple kernel that fills an array with a constant value
__global__ void fillKernel(int *d_arr, int val, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_arr[idx] = val;
}

int main()
{
    const int N = 1 << 20;              // 1M elements
    const int bytes = N * sizeof(int);

    int *h_arr = nullptr;
    int *d_arr = nullptr;

    // Allocate host memory
    h_arr = (int*)malloc(bytes);
    if (!h_arr) {
        std::cerr << "Failed to allocate host memory.\n";
        return -1;
    }

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_arr, bytes);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory: " << cudaGetErrorString(err) << "\n";
        free(h_arr);
        return -1;
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch kernel on default stream (stream 0)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    fillKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, 42, N);

    // Immediately after launch, CPU can perform other work
    std::cout << "Kernel launched asynchronously. CPU is free to do other work.\n";

    // Simulate CPU work by performing a large computation
    long long cpu_counter = 0;
    for (int i = 0; i < 1e8; ++i) {
        cpu_counter += i;
        if (i % 20000000 == 0) {
            std::cout << "CPU work progress: step " << i << "\n";
        }
    }
    std::cout << "CPU finished work. Counter = " << cpu_counter << "\n";

    // Record the stop event after CPU work
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time on GPU
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Elapsed time between kernel launch and CPU completion: " << ms << " ms\n";

    // Synchronize to ensure kernel has finished
    cudaDeviceSynchronize();

    // Copy result back to host
    err = cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy data back: " << cudaGetErrorString(err) << "\n";
    } else {
        std::cout << "First 5 elements of array after GPU kernel:\n";
        for (int i = 0; i < 5; ++i)
            std::cout << h_arr[i] << " ";
        std::cout << "\n";
    }

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}
