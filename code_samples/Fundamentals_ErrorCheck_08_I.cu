```cuda
/*
Aim of the program:
Time your program with and without `cudaDeviceSynchronize` to see the performance overhead of synchronization.

Thought process and design of the program:
1. The main goal is to illustrate how `cudaDeviceSynchronize()` affects the overall execution time of a CUDA program.
   We want to measure two things:
   a) The time it takes from launching a kernel until we immediately return control to the host (without waiting).
   b) The time it takes from launching the kernel until the host is guaranteed that the kernel has finished executing
      (after `cudaDeviceSynchronize()`).
   The difference between these two times shows the overhead introduced by the synchronization call.

2. To keep the example simple yet meaningful, we use a trivial kernel (`dummyKernel`) that performs a per-element
   addition on a large array. The kernel itself is fast, so the measured overhead is dominated by the cost of the
   synchronization rather than the kernel work.

3. Host timing is performed using `std::chrono::high_resolution_clock`. We record timestamps:
   - before the kernel launch (start)
   - immediately after the kernel launch (afterLaunch)
   - after calling `cudaDeviceSynchronize()` (afterSync)
   The elapsed times are computed as:
   * timeWithoutSync  = afterLaunch - start
   * timeWithSync     = afterSync - start
   * syncOverhead     = afterSync - afterLaunch

4. For completeness, we also measure the actual kernel execution time using CUDA events. This shows that the
   kernel itself runs for a very short period and that the large portion of the difference comes from the host-side
   waiting for the GPU.

5. Error checking is performed via a helper macro `CHECK_CUDA(call)` to ensure that all CUDA API calls
   succeed and to print meaningful error messages.

6. The program allocates memory on both host and device, copies data, runs the kernel, and finally frees resources.

7. The output prints three timings:
   - Host time without synchronization
   - Host time with synchronization
   - Overhead of `cudaDeviceSynchronize`
   It also prints the kernel execution time measured by CUDA events.

This structure allows anyone to see how much extra time the host must wait due to explicit synchronization
and demonstrates the typical overhead involved.
*/

#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl;          \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Simple kernel that adds 1.0f to each element of the array
__global__ void dummyKernel(float* a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] += 1.0f;
    }
}

int main() {
    const int N = 1 << 24;          // ~16 million elements
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    float* h_a = new float[N];
    for (int i = 0; i < N; ++i) h_a[i] = 0.0f;

    // Allocate device memory
    float* d_a = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_a, N * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));

    // ==========================
    // Timing without cudaDeviceSynchronize
    // ==========================
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    dummyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, N);

    auto afterLaunch = std::chrono::high_resolution_clock::now();

    // Now synchronize and measure the overhead
    CHECK_CUDA(cudaDeviceSynchronize());
    auto afterSync = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> timeWithoutSync = afterLaunch - start;
    std::chrono::duration<double, std::milli> timeWithSync = afterSync - start;
    std::chrono::duration<double, std::milli> syncOverhead = afterSync - afterLaunch;

    // ==========================
    // Timing kernel execution using CUDA events
    // ==========================
    cudaEvent_t eventStart, eventEnd;
    CHECK_CUDA(cudaEventCreate(&eventStart));
    CHECK_CUDA(cudaEventCreate(&eventEnd));

    CHECK_CUDA(cudaEventRecord(eventStart, 0));
    dummyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, N);
    CHECK_CUDA(cudaEventRecord(eventEnd, 0));
    CHECK_CUDA(cudaEventSynchronize(eventEnd));

    float kernelTimeMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernelTimeMs, eventStart, eventEnd));

    // ==========================
    // Copy result back to host
    // ==========================
    CHECK_CUDA(cudaMemcpy(h_a, d_a, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "Host time without cudaDeviceSynchronize: " << timeWithoutSync.count() << " ms" << std::endl;
    std::cout << "Host time with cudaDeviceSynchronize:     " << timeWithSync.count() << " ms" << std::endl;
    std::cout << "Synchronization overhead:                 " << syncOverhead.count() << " ms" << std::endl;
    std::cout << "Kernel execution time (CUDA event):        " << kernelTimeMs << " ms" << std::endl;

    // Cleanup
    delete[] h_a;
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaEventDestroy(eventStart));
    CHECK_CUDA(cudaEventDestroy(eventEnd));

    return 0;
}
```