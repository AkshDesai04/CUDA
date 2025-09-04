```cuda
/*
Demonstrate that the callback function is executed asynchronously from the host thread, and only after the kernel finishes.

Thinking:
- The goal is to show that a callback set via cudaLaunchHostFunc runs after a kernel finishes,
  but is scheduled asynchronously relative to the host thread that launched the kernel.
- To illustrate this, the program will:
  1. Record the start time.
  2. Launch a simple kernel that performs a busy-wait loop to emulate work.
  3. Immediately after launching the kernel, schedule a host callback with cudaLaunchHostFunc.
  4. The main host thread will continue executing and perform some trivial work (printing numbers).
  5. The callback will print a message, including the elapsed time since the start and the thread ID
     that executed it. This demonstrates that the callback runs after the kernel, but not on the
     original host thread (it runs on a runtime-managed thread).
  6. We also show that the host thread can continue working while the kernel is executing.
- The callback function is defined to accept a void* argument, which we use to pass the start
  timestamp. We use chrono to compute elapsed time.
- Error checking is performed after CUDA API calls.
- The kernel performs a simple loop to consume time; __syncthreads is used to ensure all threads
  are synchronized before exiting, providing a deterministic finish point.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <thread>

#define CHECK_CUDA(call)                                        \
    do {                                                        \
        cudaError_t err = (call);                               \
        if (err != cudaSuccess) {                               \
            std::cerr << "CUDA error in " << __FILE__ << ":"    \
                      << __LINE__ << " (" << #call << "): "     \
                      << cudaGetErrorString(err) << std::endl;  \
            std::exit(EXIT_FAILURE);                            \
        }                                                       \
    } while (0)

// Simple kernel that performs a busy wait loop
__global__ void busyKernel(int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < iterations; ++i) {
        // Busy wait; nothing to do
        __syncthreads(); // Ensure all threads stay in sync
    }
}

// Callback function executed after kernel completes
void CUDART_CB hostCallback(void *userData) {
    auto start = *reinterpret_cast<std::chrono::steady_clock::time_point*>(userData);
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    std::cout << "[Callback] Executed after kernel finish. "
              << "Elapsed time: " << ms << " ms. "
              << "Thread ID: " << std::this_thread::get_id() << std::endl;
}

int main() {
    // Record start time
    auto start = std::chrono::steady_clock::now();

    // Use a single stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Launch the kernel asynchronously
    int iterations = 1000000;
    int threadsPerBlock = 256;
    int blocks = 64;
    std::cout << "Launching kernel asynchronously..." << std::endl;
    busyKernel<<<blocks, threadsPerBlock, 0, stream>>>(iterations);
    CHECK_CUDA(cudaGetLastError());

    // Schedule the callback to run after kernel completion
    std::cout << "Scheduling callback via cudaLaunchHostFunc..." << std::endl;
    CHECK_CUDA(cudaLaunchHostFunc(stream, hostCallback, &start));

    // Main host thread continues doing other work
    std::cout << "Host thread continues doing work (printing numbers 0-5)... "
              << "Thread ID: " << std::this_thread::get_id() << std::endl;
    for (int i = 0; i < 6; ++i) {
        std::cout << "Host work: " << i << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Simulate work
    }

    // Wait for all operations in the stream to complete
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::cout << "All operations complete. Program exiting." << std::endl;

    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}
```