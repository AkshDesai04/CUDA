```cuda
/*
Aim of the program:
Time your program with and without `cudaDeviceSynchronize` to see the performance overhead of synchronization.

Thinking:
- The goal is to demonstrate how the call to `cudaDeviceSynchronize` adds overhead to a CUDA program.
- A simple kernel is used that performs a modest amount of work (e.g., populating an array) so that the kernel execution time is non‑zero but not dominated by data transfer.
- Timing is performed on the host using `std::chrono::high_resolution_clock` because it is easy to understand and does not require GPU events.
- We take three timestamps:
  1. `t_before_launch` – just before the kernel is launched.
  2. `t_after_launch`  – immediately after the kernel launch.
  3. `t_after_sync`    – after a `cudaDeviceSynchronize()` call.
- The difference `t_after_launch - t_before_launch` gives the time taken to launch the kernel (essentially the overhead of enqueuing the kernel, which is very small).
- The difference `t_after_sync - t_before_launch` gives the total time from launch to completion, which includes both the kernel execution time and the overhead of `cudaDeviceSynchronize`.
- By subtracting the two durations we obtain the pure synchronization overhead.
- Error checking is performed after each CUDA API call to ensure correctness.
- The program prints:
  * Kernel launch overhead
  * Synchronization overhead
  * Total kernel execution + sync time
- Finally, the result is validated to ensure the kernel actually ran correctly.

This program is self‑contained and can be compiled with `nvcc`:
    nvcc -O2 -std=c++11 sync_overhead.cu -o sync_overhead
and run with:
    ./sync_overhead
*/

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cassert>

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in file '" << __FILE__ << "' in line " \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Simple kernel that writes index to the array
__global__ void fill_array(int *d_arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_arr[idx] = idx;
}

int main()
{
    const int N = 1 << 20;               // 1M elements
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    std::vector<int> h_arr(N);
    // Allocate device memory
    int *d_arr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_arr, N * sizeof(int)));

    // Record time before kernel launch
    auto t_before_launch = std::chrono::high_resolution_clock::now();

    // Launch kernel
    fill_array<<<blocks, threadsPerBlock>>>(d_arr, N);

    // Record time immediately after kernel launch
    auto t_after_launch = std::chrono::high_resolution_clock::now();

    // Synchronize to ensure kernel has finished
    CHECK_CUDA(cudaDeviceSynchronize());

    // Record time after synchronization
    auto t_after_sync = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_arr.data(), d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Validate results
    bool valid = true;
    for (int i = 0; i < N; ++i) {
        if (h_arr[i] != i) {
            valid = false;
            std::cerr << "Error at index " << i << ": expected " << i
                      << ", got " << h_arr[i] << "\n";
            break;
        }
    }
    if (valid) {
        std::cout << "Kernel execution validated successfully.\n";
    }

    // Compute durations
    std::chrono::duration<double, std::micro> launch_time = t_after_launch - t_before_launch;
    std::chrono::duration<double, std::micro> sync_time = t_after_sync - t_after_launch;
    std::chrono::duration<double, std::micro> total_time = t_after_sync - t_before_launch;

    std::cout << "Kernel launch overhead:          " << launch_time.count() << " us\n";
    std::cout << "Synchronization overhead:       " << sync_time.count() << " us\n";
    std::cout << "Total kernel + sync time:        " << total_time.count() << " us\n";

    // Clean up
    CHECK_CUDA(cudaFree(d_arr));
    return 0;
}
```