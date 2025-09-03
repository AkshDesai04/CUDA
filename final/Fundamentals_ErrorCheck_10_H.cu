/*
Aim: Add a `reset()` method that allows the timer to be reused without creating a new object.

Thought process:
- The requirement is to allow a timer object to be reused multiple times without needing to instantiate a new object each time.
- In CUDA, the typical way to time GPU work is via cudaEvent_t start and stop events.
- A simple wrapper class GPUTimer will be created. It will:
  * Create two events (start, stop) in its constructor.
  * Provide startTimer() and stopTimer() methods to record events.
  * Provide elapsedTime() to compute the elapsed milliseconds between the recorded events.
  * Provide a reset() method that destroys the existing events and recreates them, effectively clearing the timer state.
- The reset() method must also reset the running flag to false.
- The destructor will clean up the events.
- A small demo kernel and main function will illustrate using the timer, calling reset(), and timing again.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple CUDA kernel used for timing demo
__global__ void dummyKernel(int *a) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    a[idx] = idx * idx;
}

// GPUTimer class encapsulates CUDA event-based timing
class GPUTimer {
private:
    cudaEvent_t start;
    cudaEvent_t stop;
    bool running;

public:
    // Constructor: create events
    GPUTimer() : running(false) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    // Destructor: destroy events
    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Start the timer
    void startTimer() {
        if (!running) {
            cudaEventRecord(start, 0);
            running = true;
        }
    }

    // Stop the timer
    void stopTimer() {
        if (running) {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            running = false;
        }
    }

    // Return elapsed time in milliseconds
    float elapsedTime() const {
        float ms = 0.0f;
        if (!running) {
            cudaEventElapsedTime(&ms, start, stop);
        }
        return ms;
    }

    // Reset the timer for reuse
    void reset() {
        // Destroy old events and create new ones
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        running = false;
    }
};

int main() {
    const int N = 1 << 20;  // 1M elements
    int *dev_a;
    cudaMalloc(&dev_a, N * sizeof(int));

    GPUTimer timer;

    // First timing run
    timer.startTimer();
    dummyKernel<<<(N + 255) / 256, 256>>>(dev_a);
    timer.stopTimer();
    printf("Kernel elapsed time: %.3f ms\n", timer.elapsedTime());

    // Reset timer for next measurement
    timer.reset();

    // Second timing run
    timer.startTimer();
    dummyKernel<<<(N + 255) / 256, 256>>>(dev_a);
    timer.stopTimer();
    printf("Kernel elapsed time after reset: %.3f ms\n", timer.elapsedTime());

    cudaFree(dev_a);
    return 0;
}