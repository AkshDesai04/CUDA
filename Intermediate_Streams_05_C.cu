```cuda
/*
Demonstrate that the callback function is executed asynchronously from the host thread,
and only after the kernel finishes.

Thinking process:
1. We need a CUDA kernel that takes a noticeable amount of time so we can observe
   the host thread doing something else concurrently. A simple busy‑wait loop will
   suffice. The kernel will run on a separate CUDA stream.
2. CUDA provides `cudaStreamAddCallback` which queues a host function to be called
   after all preceding operations in the stream have finished. The callback runs on
   a different host thread (managed by the CUDA runtime) and is therefore
   asynchronous relative to the thread that added it.
3. To demonstrate this, the main thread will:
   - create a stream,
   - launch the kernel on that stream,
   - register a callback,
   - then immediately perform some host work (printing messages and sleeping).
4. The callback will:
   - print a message,
   - record the current time,
   - set an atomic flag so the main thread knows it has finished.
5. The main thread will wait on that flag after the host work is done, ensuring
   it only continues once the callback has executed.
6. By printing timestamps before and after the kernel launch, during the host
   work, and inside the callback, we can clearly see that:
   * The host continues executing immediately after launching the kernel,
   * The callback is invoked only after the kernel has finished,
   * The callback runs on a separate thread (different thread ID from the main thread).
*/

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <iomanip>

// Kernel that performs a busy wait to simulate work
__global__ void busyKernel(int iterations)
{
    // Simple loop; not optimized for real work, just to consume time
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long counter = 0;
    for (int i = 0; i < iterations; ++i)
    {
        counter += tid;
    }
    // Prevent compiler from optimizing away
    if (counter == 0xFFFFFFFFFFFFFFFFULL)
        printf("Impossible");
}

// Structure to pass to callback
struct CallbackContext
{
    std::atomic<bool> finished;
    std::chrono::steady_clock::time_point start;
};

void CUDART_CB kernelCallback(cudaStream_t stream, cudaError_t status, void *userData)
{
    CallbackContext* ctx = static_cast<CallbackContext*>(userData);
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - ctx->start).count();

    std::cout << "[Callback] Thread ID: " << std::this_thread::get_id()
              << " | Callback executed after kernel\n";
    std::cout << "[Callback] Elapsed time since kernel launch: "
              << elapsed << " ms\n";
    ctx->finished.store(true, std::memory_order_release);
}

int main()
{
    // Create a CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Context for the callback
    CallbackContext ctx;
    ctx.finished.store(false, std::memory_order_relaxed);
    ctx.start = std::chrono::steady_clock::now();

    // Parameters for the kernel
    int threadsPerBlock = 256;
    int blocks = 64;
    int iterations = 100000000; // Adjust to make kernel run for ~1-2 seconds

    std::cout << "[Host] Launching kernel\n";
    busyKernel<<<blocks, threadsPerBlock, 0, stream>>>(iterations);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Add callback to the stream
    std::cout << "[Host] Adding callback to stream\n";
    err = cudaStreamAddCallback(stream, kernelCallback, &ctx, 0);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to add callback: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Host thread continues doing other work
    std::cout << "[Host] Continuing work on host thread. Thread ID: " << std::this_thread::get_id() << "\n";
    std::cout << "[Host] Sleeping for 1 second to simulate work...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "[Host] Host work done. Waiting for callback to finish...\n";

    // Wait until callback signals completion
    while (!ctx.finished.load(std::memory_order_acquire))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "[Host] Callback has finished. Cleaning up.\n";

    // Synchronize to ensure all stream work is complete
    cudaStreamSynchronize(stream);

    // Destroy stream
    cudaStreamDestroy(stream);

    return 0;
}
```