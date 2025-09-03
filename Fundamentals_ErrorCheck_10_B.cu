```cuda
/********************************************************************
 * Make the `stop()` method non-blocking by not calling `cudaEventSynchronize`. 
 * The user must call a separate `synchronize()` method before `elapsed_ms()` will be accurate. 
 *
 * Thinking:
 *   The original design of a CUDA timer often uses cudaEventRecord to capture start
 *   and stop times, then synchronizes the stop event before computing elapsed
 *   time to ensure the GPU has finished all work. This can serialize GPU
 *   execution if called from a host thread. To make the timer more flexible,
 *   we remove the blocking call from `stop()` and expose a separate
 *   `synchronize()` method. This allows users to continue launching kernels
 *   or doing other work after `stop()` and only synchronize when they
 *   actually need the elapsed time. The implementation follows the usual
 *   CUDA event pattern: two events are created, one for start and one for
 *   stop. `start()` records the start event. `stop()` records the stop
 *   event without synchronizing. `synchronize()` waits for the stop
 *   event to complete. `elapsed_ms()` first checks if the stop event has
 *   been synchronized; if not, it calls `cudaEventSynchronize` internally
 *   (or returns an error). For simplicity, `elapsed_ms()` will simply
 *   perform a blocking synchronization to guarantee accuracy, but users
 *   can call `synchronize()` explicitly if they wish to combine multiple
 *   timing calls or reuse the timer. The class also cleans up events
 *   in its destructor. A small example in `main()` demonstrates usage. 
 ********************************************************************/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

class CUDATimer
{
public:
    CUDATimer()
    {
        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);
        synchronized_ = false;
    }

    ~CUDATimer()
    {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
    }

    // Record the start time
    void start()
    {
        cudaEventRecord(start_event_, 0);
        synchronized_ = false;
    }

    // Record the stop time non‑blocking
    void stop()
    {
        cudaEventRecord(stop_event_, 0);
        synchronized_ = false;
    }

    // Block until the stop event has finished executing
    void synchronize()
    {
        cudaEventSynchronize(stop_event_);
        synchronized_ = true;
    }

    // Return elapsed time in milliseconds. If the stop event has not
    // been synchronized yet, this will block to produce accurate result.
    float elapsed_ms()
    {
        if (!synchronized_)
        {
            cudaEventSynchronize(stop_event_);
            synchronized_ = true;
        }
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_event_, stop_event_);
        return ms;
    }

private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    bool synchronized_;
};

// A trivial kernel for demonstration purposes
__global__ void dummyKernel()
{
    // Do nothing
}

int main()
{
    const int threads = 256;
    const int blocks  = 64;

    CUDATimer timer;

    // Example: measure kernel execution time without blocking at stop()
    timer.start();
    dummyKernel<<<blocks, threads>>>();
    timer.stop();                 // non‑blocking stop
    // ... do other work here if desired ...
    timer.synchronize();          // ensure kernel finished before timing

    float elapsed = timer.elapsed_ms();
    printf("Elapsed time: %f ms\n", elapsed);

    return 0;
}
```