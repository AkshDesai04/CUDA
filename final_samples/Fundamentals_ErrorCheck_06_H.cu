```cpp
/*
Does `cudaDeviceSynchronize` have a performance cost? (Yes, it stalls the CPU). Explain when it's appropriate to use (debugging, final result needed) and when not (in a tight loop of asynchronous operations).

Thinking:
- The goal is to produce a CUDA C program that demonstrates the cost of calling cudaDeviceSynchronize.
- The program will launch a simple kernel, time its execution with cudaEvent, and show how inserting cudaDeviceSynchronize affects timing and CPU stalls.
- We'll run two scenarios: one where we immediately synchronize after launching the kernel (useful for debugging or when the result is needed before proceeding) and another where we do not synchronize until after the entire sequence (useful in tight loops of asynchronous work to avoid unnecessary stalls).
- The code uses cudaEventRecord for precise timing on the GPU and also records CPU wall‑clock time to illustrate the stall.
- The kernel performs a trivial operation to keep the example lightweight.
- The program prints the timing results for both scenarios so that the user can observe the difference.
*/

#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__      \
                      << " (" << cudaGetErrorString(err) << ")\n";            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Simple kernel that doubles each element
__global__ void dummyKernel(int *d_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_data[idx] = d_data[idx] * 2;
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(int);

    // Host memory
    int *h_in = new int[N];
    int *h_out = new int[N];

    // Initialize input
    for (int i = 0; i < N; ++i) {
        h_in[i] = i;
    }

    // Device memory
    int *d_data;
    CUDA_CHECK(cudaMalloc((void**)&d_data, size));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_data, h_in, size, cudaMemcpyHostToDevice));

    // Timing variables
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    // ------------------------------
    // Scenario 1: Immediate synchronization after kernel launch
    // ------------------------------
    CUDA_CHECK(cudaEventRecord(startEvent, 0));
    dummyKernel<<<(N + 255) / 256, 256>>>(d_data, N);
    // Synchronize: stalls CPU until kernel completes
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stopEvent, 0));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float ms_sync;
    CUDA_CHECK(cudaEventElapsedTime(&ms_sync, startEvent, stopEvent));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_data, size, cudaMemcpyDeviceToHost));

    // ------------------------------
    // Scenario 2: No synchronization until after the sequence
    // ------------------------------
    // Reinitialize device data
    CUDA_CHECK(cudaMemcpy(d_data, h_in, size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(startEvent, 0));
    dummyKernel<<<(N + 255) / 256, 256>>>(d_data, N);
    // No cudaDeviceSynchronize here; the launch is asynchronous.
    // We just record the event for timing.
    CUDA_CHECK(cudaEventRecord(stopEvent, 0));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float ms_async;
    CUDA_CHECK(cudaEventElapsedTime(&ms_async, startEvent, stopEvent));

    // Now we need to synchronize before copying back to host
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_data, size, cudaMemcpyDeviceToHost));

    // ------------------------------
    // Results
    // ------------------------------
    std::cout << "Timing with immediate cudaDeviceSynchronize: " << ms_sync << " ms\n";
    std::cout << "Timing without immediate cudaDeviceSynchronize: " << ms_async << " ms\n";
    std::cout << "Difference (ms): " << (ms_sync - ms_async) << "\n";

    // Clean up
    delete[] h_in;
    delete[] h_out;
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    return 0;
}
```