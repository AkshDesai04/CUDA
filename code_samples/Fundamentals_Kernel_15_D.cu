```cuda
/* 
Aim: Write a kernel that has a very long but finite loop. Call it without 
`cudaDeviceSynchronize`. The host program finishes almost instantly. Now add 
`cudaDeviceSynchronize`. The host program now waits for the kernel to complete.

Thinking:
The task is to demonstrate the effect of synchronizing after launching a 
CUDA kernel. A kernel that contains a large, deterministic loop will run for
a noticeable amount of time on the GPU, but the host will not block unless
explicitly told to wait. To make the example clear, we perform two 
executions in the same program:

1. **Without Synchronization**  
   - Allocate a single-element array on the device.  
   - Launch `longLoopKernel` with a very large iteration count (e.g. 1e9).  
   - Do *not* call `cudaDeviceSynchronize()`.  
   - Measure and print the host execution time.  
   - Since no synchronization occurs, the host should finish almost 
     immediately, regardless of the kernel still running.

2. **With Synchronization**  
   - Launch the same kernel again.  
   - Call `cudaDeviceSynchronize()` to block the host until the kernel has 
     finished.  
   - Measure and print the host execution time.  
   - The host should now wait for the kernel to finish, resulting in a
     noticeably longer elapsed time.

To avoid the compiler removing the loop (because the result is unused), 
the kernel writes the final sum into the output array. The sum itself is
not used after kernel completion, so the compiler cannot eliminate the
loop. We also use a single thread to keep the example simple.

The program uses `std::chrono` to measure host time for both runs, 
printing the elapsed milliseconds. This gives a clear visual difference
between the two cases.

Note: In a real application you would normally free GPU memory and
check for errors, but for brevity this example focuses solely on the
synchronization effect.
*/

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// Simple error-checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in file '" << __FILE__                 \
                      << "' in line " << __LINE__ << ": "                   \
                      << cudaGetErrorString(err) << std::endl;              \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel that performs a long, finite loop
__global__ void longLoopKernel(long long N, long long *output) {
    // Only thread 0 does the work to keep the example simple
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        long long sum = 0;
        for (long long i = 0; i < N; ++i) {
            sum += i;          // Use sum to prevent compiler elimination
        }
        *output = sum;          // Store result to avoid unused variable
    }
}

int main() {
    const long long LOOP_COUNT = 1000000000LL; // 1e9 iterations
    const int THREADS = 1;
    const int BLOCKS = 1;

    // Allocate device memory for output
    long long *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(long long)));

    // ---------- Run 1: Without cudaDeviceSynchronize ----------
    {
        std::cout << "=== Run 1: Launch kernel WITHOUT synchronization ===" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        // Launch kernel
        longLoopKernel<<<BLOCKS, THREADS>>>(LOOP_COUNT, d_output);

        // No synchronization: host continues immediately
        auto mid = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> host_time_no_sync =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(mid - start);
        std::cout << "Host elapsed time after kernel launch (no sync): "
                  << host_time_no_sync.count() << " ms" << std::endl;

        // Host does other work (simulated by a small sleep)
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Note: Not copying or freeing device memory here, to keep the
        // kernel still running while host exits. The program will exit
        // and OS will clean up GPU resources.
    }

    // Wait a short time before the second run to ensure previous kernel is still running
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // ---------- Run 2: With cudaDeviceSynchronize ----------
    {
        std::cout << "\n=== Run 2: Launch kernel WITH synchronization ===" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        // Launch kernel
        longLoopKernel<<<BLOCKS, THREADS>>>(LOOP_COUNT, d_output);

        // Now wait for the kernel to finish
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> host_time_sync =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
        std::cout << "Host elapsed time after cudaDeviceSynchronize: "
                  << host_time_sync.count() << " ms" << std::endl;

        // Optionally copy result back (also a sync, but not needed for demo)
        // long long host_output;
        // CUDA_CHECK(cudaMemcpy(&host_output, d_output, sizeof(long long), cudaMemcpyDeviceToHost));
        // std::cout << "Kernel result: " << host_output << std::endl;

        // Clean up device memory
        CUDA_CHECK(cudaFree(d_output));
    }

    return 0;
}
```