/*
Aim of the program: Compare the timing from CUDA events to timing with `std::chrono::high_resolution_clock`. Explain why CUDA events are more accurate for timing GPU operations.

Thinking:
- The goal is to demonstrate that CUDA events provide a more precise measurement of GPU execution time compared to using the host clock (`std::chrono`). 
- CUDA events are recorded on the GPU, so the elapsed time excludes host overhead, CPU scheduling, and any driver latency that may affect a host timer. 
- `std::chrono` measures wall‑clock time on the CPU, which must wait for the GPU kernel to finish (via `cudaDeviceSynchronize()`) before it can compute the elapsed time. This introduces additional variability due to CPU context switches, thread scheduling, and the time taken to launch the kernel. 
- A simple GPU kernel (vector addition) is used. The program runs the kernel twice: once timed with CUDA events and once timed with `std::chrono`. Both timings include a `cudaDeviceSynchronize()` to ensure the kernel has finished before measurement is taken. 
- After both measurements, the program prints the times and explains the observed difference. 
- The code is self‑contained, compiles to a single `.cu` file, and includes minimal error checking for clarity. 
- The kernel size and launch configuration are chosen to make the GPU work noticeable while keeping the example simple. 
*/

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// Simple kernel: fill array with i*i
__global__ void dummyKernel(int *d_arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_arr[idx] = idx * idx;
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(err)                                                     \
    do {                                                                    \
        cudaError_t err__ = (err);                                          \
        if (err__ != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " - " << cudaGetErrorString(err__) << std::endl;  \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                   \
    } while (0)

int main()
{
    const int N = 1 << 24;            // 16 million elements
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host and device memory
    int *h_arr = new int[N];
    int *d_arr;
    CUDA_CHECK(cudaMalloc(&d_arr, N * sizeof(int)));

    // Warm up the GPU (optional)
    dummyKernel<<<blocks, threadsPerBlock>>>(d_arr, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ---------- Timing with CUDA events ---------- */
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    CUDA_CHECK(cudaEventRecord(startEvent, 0));
    dummyKernel<<<blocks, threadsPerBlock>>>(d_arr, N);
    CUDA_CHECK(cudaEventRecord(stopEvent, 0));

    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float gpuTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTimeMs, startEvent, stopEvent));

    /* ---------- Timing with std::chrono ---------- */
    auto t1 = std::chrono::high_resolution_clock::now();
    dummyKernel<<<blocks, threadsPerBlock>>>(d_arr, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> hostDurationMs = t2 - t1;

    /* ---------- Output ---------- */
    std::cout << "GPU time measured by CUDA events:  " << gpuTimeMs << " ms\n";
    std::cout << "GPU time measured by std::chrono: " << hostDurationMs.count() << " ms\n";

    /* ---------- Explanation ---------- */
    std::cout << "\nExplanation:\n";
    std::cout << "CUDA events record timestamps directly on the GPU device.\n";
    std::cout << "They measure only the time the GPU spends executing the kernel\n";
    std::cout << "and are unaffected by CPU scheduling, driver overhead, or\n";
    std::cout << "other host‑side activities. std::chrono, on the other hand,\n";
    std::cout << "measures wall‑clock time on the CPU. Even with a device\n";
    std::cout << "synchronization call, the host clock also includes the time\n";
    std::cout << "taken to launch the kernel, context switching, and other\n";
    std::cout << "non‑deterministic host operations, making it less accurate\n";
    std::cout << "for measuring GPU execution alone.\n";

    // Clean up
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(d_arr));
    delete[] h_arr;

    return 0;
}
