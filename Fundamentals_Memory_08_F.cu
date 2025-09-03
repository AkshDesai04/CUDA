/*
Wrap the event timer logic in a C++ class `GpuTimer` for convenience.

Thinking:
The goal is to provide a simple, reusable C++ class that encapsulates CUDA event creation, timing, and destruction logic. The class should:
1. Create two CUDA events (start and stop) upon construction.
2. Provide a `startTimer()` method that records the start event.
3. Provide a `stopTimer()` method that records the stop event, synchronizes the device to ensure all preceding work is finished, and optionally records the elapsed time.
4. Provide an `elapsedTime()` method that returns the elapsed time between the start and stop events in milliseconds.
5. Clean up the CUDA events in the destructor.
6. Optionally expose a `reset()` method to reuse the timer without needing to destroy/recreate the events.

The example will include a trivial kernel launch and demonstrate timing its execution using the `GpuTimer` class. The code is self‑contained, written in a single .cu file, and uses only the standard CUDA runtime API and C++11 features.

Key points for correctness:
- CUDA event creation must succeed; we will check for errors.
- The device must be synchronized after the stop event to guarantee accurate timing.
- The destructor must release the events to avoid memory leaks.
- The example kernel will be very small but will still show the timer working.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

/**
 * @brief Simple helper macro to check CUDA API calls.
 */
#define CHECK_CUDA(call)                                                    \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in file '" << __FILE__                 \
                      << "' in line " << __LINE__ << ": "                   \
                      << cudaGetErrorString(err) << std::endl;              \
            std::terminate();                                               \
        }                                                                    \
    } while (0)

/**
 * @brief A small utility class that wraps CUDA event timing.
 *
 * The class handles creation, destruction, and timing of CUDA events
 * to measure elapsed GPU time. It is intentionally lightweight and
 * does not depend on any external libraries beyond the CUDA runtime.
 */
class GpuTimer {
public:
    /**
     * @brief Construct a new GpuTimer object.
     *
     * Two CUDA events are created (start and stop). They are
     * initialized with default flags.
     */
    GpuTimer() {
        CHECK_CUDA(cudaEventCreate(&startEvent_));
        CHECK_CUDA(cudaEventCreate(&stopEvent_));
    }

    /**
     * @brief Destroy the GpuTimer object.
     *
     * Both CUDA events are destroyed. No exceptions are thrown.
     */
    ~GpuTimer() {
        cudaEventDestroy(startEvent_);
        cudaEventDestroy(stopEvent_);
    }

    /**
     * @brief Record the start event.
     *
     * This marks the beginning of the timed region. The event is
     * recorded on the default stream.
     */
    void startTimer() {
        CHECK_CUDA(cudaEventRecord(startEvent_, 0));
    }

    /**
     * @brief Record the stop event and synchronize the device.
     *
     * After recording the stop event, the device is synchronized
     * to ensure that all preceding work has completed before
     * we query the elapsed time.
     */
    void stopTimer() {
        CHECK_CUDA(cudaEventRecord(stopEvent_, 0));
        CHECK_CUDA(cudaEventSynchronize(stopEvent_));
    }

    /**
     * @brief Get the elapsed time in milliseconds.
     *
     * @return float Elapsed time between start and stop events.
     */
    float elapsedTime() const {
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, startEvent_, stopEvent_));
        return ms;
    }

    /**
     * @brief Reset the timer by recreating the events.
     *
     * This is optional but useful if you want to reuse the same
     * GpuTimer instance for multiple timed operations without
     * creating a new object each time.
     */
    void reset() {
        cudaEventDestroy(startEvent_);
        cudaEventDestroy(stopEvent_);
        CHECK_CUDA(cudaEventCreate(&startEvent_));
        CHECK_CUDA(cudaEventCreate(&stopEvent_));
    }

private:
    cudaEvent_t startEvent_;
    cudaEvent_t stopEvent_;
};

/**
 * @brief A trivial CUDA kernel used for demonstration.
 *
 * It simply writes a value to the first element of an array.
 */
__global__ void dummyKernel(int* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        data[0] = 42;
    }
}

int main() {
    // Allocate device memory for a single integer.
    int* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, sizeof(int)));

    GpuTimer timer;

    // Launch the kernel and time its execution.
    timer.startTimer();
    dummyKernel<<<1, 1>>>(d_data);
    timer.stopTimer();

    // Copy result back to host to ensure kernel finished.
    int h_data = 0;
    CHECK_CUDA(cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));

    // Output the timing result and kernel output.
    std::cout << "Kernel result: " << h_data << std::endl;
    std::cout << "Elapsed GPU time: " << timer.elapsedTime() << " ms" << std::endl;

    // Clean up
    CHECK_CUDA(cudaFree(d_data));
    return 0;
}
