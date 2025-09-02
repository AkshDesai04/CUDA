/*
Aim: Make the `stop()` method non-blocking by not calling `cudaEventSynchronize`. The user must call a separate `synchronize()` method before `elapsed_ms()` will be accurate.

Thinking:
To satisfy the requirement, I will create a simple CUDA timer wrapper that uses `cudaEvent_t` objects for timing.  
The wrapper will expose the following public methods:

1. `start()` – records a start event.
2. `stop()`  – records a stop event but **does not** wait for the event to finish.  This makes the stop call non‑blocking.
3. `synchronize()` – calls `cudaEventSynchronize(stopEvent)` to ensure that the stop event has finished.  The user must invoke this before requesting the elapsed time.
4. `elapsed_ms()` – calls `cudaEventElapsedTime(&ms, startEvent, stopEvent)` to return the elapsed milliseconds.  If `synchronize()` was not called, the result may be inaccurate.

Internally the class will create two events (`startEvent` and `stopEvent`) with default flags.  
The destructor will destroy the events.  

A minimal `main()` function is provided to demonstrate the usage: start the timer, launch a trivial kernel, stop the timer, synchronize, then query the elapsed time.

The code is written in plain CUDA C (`.cu` file) and can be compiled with `nvcc`.  No additional libraries or dependencies are required.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel for demonstration purposes
__global__ void dummyKernel(int *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_arr[idx] = d_arr[idx] * 2;
}

// CUDA Timer class
class CudaTimer
{
public:
    CudaTimer()
    {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
    }

    ~CudaTimer()
    {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    // Record the start event
    void start()
    {
        cudaEventRecord(startEvent, 0);
    }

    // Record the stop event (non‑blocking)
    void stop()
    {
        cudaEventRecord(stopEvent, 0);
        // Note: No cudaEventSynchronize here – making stop non‑blocking
    }

    // User must call this before elapsed_ms() for accurate timing
    void synchronize()
    {
        cudaEventSynchronize(stopEvent);
    }

    // Return elapsed time in milliseconds
    float elapsed_ms()
    {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        return ms;
    }

private:
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
};

int main()
{
    const int N = 1 << 20;
    int *h_arr = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i)
        h_arr[i] = i;

    int *d_arr;
    cudaMalloc((void**)&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    CudaTimer timer;
    timer.start();

    // Launch kernel
    dummyKernel<<<(N + 255) / 256, 256>>>(d_arr, N);

    timer.stop();

    // At this point, the kernel might still be running; stop() was non‑blocking
    // We synchronize before querying elapsed time
    timer.synchronize();

    float elapsed = timer.elapsed_ms();
    printf("Kernel execution time: %f ms\n", elapsed);

    // Clean up
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}
