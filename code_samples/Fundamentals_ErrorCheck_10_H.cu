```cuda
/********************************************************************
 * Aim of the program: 
 * Add a `reset()` method that allows the timer to be reused without creating a new object.
 *
 * My thoughts:
 * - The program is a small CUDA timer utility that measures GPU kernel execution time using
 *   cudaEvent.  The standard pattern is to create two events, startEvent and stopEvent,
 *   record them around the code to measure, and then compute elapsed time with
 *   cudaEventElapsedTime.  In many applications the timer object is instantiated
 *   once and reused, but the original implementation did not provide a way to
 *   clear the state between measurements.  A `reset()` method is needed so that
 *   the same timer instance can be reused without having to allocate a new one.
 *
 * - The simplest way to reset the timer is to destroy the existing events and
 *   create fresh ones.  This guarantees that any previously recorded timestamps
 *   are discarded.  An alternative would be to keep the same events and just
 *   record new timestamps, but that can lead to confusion if a user stops the
 *   timer and then starts it again without realizing the old events are still
 *   valid.  Recreating the events makes the semantics clear.
 *
 * - I will implement a GPUTimer class with the following public interface:
 *     * start()   – records startEvent.
 *     * stop()    – records stopEvent, synchronizes, and marks the timer as stopped.
 *     * elapsed() – returns elapsed time in milliseconds between start and stop.
 *     * reset()   – destroys and recreates the event objects, resetting state.
 *
 * - The destructor will clean up the events.  Error checking will be performed
 *   with a simple helper function to keep the code concise.
 *
 * - A small main function demonstrates usage: it launches a dummy kernel
 *   (an empty kernel) and measures the time, then resets the timer and runs
 *   it again, printing both measurements.
 *
 * This implementation is self-contained and can be compiled with `nvcc
 * timer.cu -o timer`.  It will print two identical measurements (close to
 * 0 ms) showing that the reset() works correctly.
 *********************************************************************/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Helper macro for error checking
#define CUDA_CHECK(call)                                    \
    do {                                                    \
        cudaError_t err = (call);                           \
        if (err != cudaSuccess) {                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

// Dummy kernel to have something to time
__global__ void dummyKernel()
{
    // Do nothing
}

class GPUTimer {
private:
    cudaEvent_t startEvent_;
    cudaEvent_t stopEvent_;
    bool running_;

    void createEvents()
    {
        CUDA_CHECK(cudaEventCreate(&startEvent_));
        CUDA_CHECK(cudaEventCreate(&stopEvent_));
        running_ = false;
    }

public:
    GPUTimer()
    {
        createEvents();
    }

    ~GPUTimer()
    {
        CUDA_CHECK(cudaEventDestroy(startEvent_));
        CUDA_CHECK(cudaEventDestroy(stopEvent_));
    }

    // Record start event
    void start()
    {
        CUDA_CHECK(cudaEventRecord(startEvent_, 0));
        running_ = true;
    }

    // Record stop event and synchronize
    void stop()
    {
        if (!running_) {
            fprintf(stderr, "Timer::stop() called without start().\n");
            return;
        }
        CUDA_CHECK(cudaEventRecord(stopEvent_, 0));
        CUDA_CHECK(cudaEventSynchronize(stopEvent_));
        running_ = false;
    }

    // Return elapsed time in milliseconds
    float elapsed()
    {
        if (running_) {
            fprintf(stderr, "Timer::elapsed() called while timer is running.\n");
            return 0.0f;
        }
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent_, stopEvent_));
        return ms;
    }

    // Reset the timer by destroying and recreating events
    void reset()
    {
        CUDA_CHECK(cudaEventDestroy(startEvent_));
        CUDA_CHECK(cudaEventDestroy(stopEvent_));
        createEvents();
    }
};

int main()
{
    // Initialize CUDA device
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    // Create a timer
    GPUTimer timer;

    // First measurement
    timer.start();
    dummyKernel<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());
    timer.stop();
    printf("Elapsed time (first run): %.3f ms\n", timer.elapsed());

    // Reset the timer
    timer.reset();

    // Second measurement
    timer.start();
    dummyKernel<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());
    timer.stop();
    printf("Elapsed time (second run after reset): %.3f ms\n", timer.elapsed());

    // Clean up
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
```