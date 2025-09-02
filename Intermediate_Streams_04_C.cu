```cpp
/*
Aim: Explain why this is more efficient than using `cudaDeviceSynchronize()` between the kernels (it doesn't stall the host CPU and only synchronizes the necessary streams).

Thinking:
To illustrate the efficiency advantage, the program demonstrates launching two simple kernels in distinct CUDA streams and synchronizing them using stream events instead of a global device synchronization. The key points are:
1. `cudaDeviceSynchronize()` blocks the host until all device work in all streams completes, forcing the CPU to wait even if only one stream requires completion. This serialises host–device interaction.
2. By launching kernels in separate streams and using `cudaEventRecord`/`cudaEventSynchronize` or `cudaStreamWaitEvent`, we enforce only the necessary dependency (e.g., kernel B depends on kernel A) while allowing other streams to continue executing.
3. The host can perform other CPU work between kernel launches because it is not blocked. The device can schedule and execute the streams concurrently, improving overall throughput.
4. The code below shows a minimal example: two kernels `kernelA` and `kernelB` are launched in streams 0 and 1 respectively. Kernel B waits on an event recorded after kernel A, ensuring correct ordering without a global device sync. The host measures elapsed time, demonstrating that the host remains responsive.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// Simple kernel A: fill array with value 1
__global__ void kernelA(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_arr[idx] = 1;
}

// Simple kernel B: add 2 to each element
__global__ void kernelB(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_arr[idx] += 2;
}

int main() {
    const int N = 1 << 20;           // 1M elements
    const int THREADS_PER_BLOCK = 256;
    const int BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    int *d_arr;
    cudaMalloc(&d_arr, N * sizeof(int));

    // Create two streams
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    // Event to signal completion of kernelA
    cudaEvent_t eventA;
    cudaEventCreate(&eventA);

    // Launch kernelA in stream0
    kernelA<<<BLOCKS, THREADS_PER_BLOCK, 0, stream0>>>(d_arr, N);
    cudaEventRecord(eventA, stream0);  // record event after kernelA

    // Launch kernelB in stream1, but make it wait for eventA
    cudaStreamWaitEvent(stream1, eventA, 0);
    kernelB<<<BLOCKS, THREADS_PER_BLOCK, 0, stream1>>>(d_arr, N);

    // Record event after kernelB
    cudaEvent_t eventB;
    cudaEventCreate(&eventB);
    cudaEventRecord(eventB, stream1);

    // Host timing
    auto start = std::chrono::high_resolution_clock::now();

    // Wait for eventB to ensure both kernels finished
    cudaEventSynchronize(eventB);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time elapsed (no cudaDeviceSynchronize): " << elapsed.count() << " s\n";

    // Verify result
    int *h_arr = new int[N];
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_arr[i] != 3) { ok = false; break; }
    }
    std::cout << "Result verification: " << (ok ? "PASS" : "FAIL") << "\n";

    // Clean up
    delete[] h_arr;
    cudaEventDestroy(eventA);
    cudaEventDestroy(eventB);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaFree(d_arr);

    return 0;
}
```