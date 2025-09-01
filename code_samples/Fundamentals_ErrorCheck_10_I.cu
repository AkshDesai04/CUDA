```cpp
/*
Make the class non-copyable by deleting the copy constructor and copy assignment operator, as `cudaEvent_t` handles cannot be trivially copied.

Thinking:
- We wrap cudaEvent_t in a small RAII class (`CudaEvent`) so that event creation and destruction are automatically managed.
- CUDA events are not copyable because they are GPU resources; copying would duplicate the handle without duplicating the underlying resource, leading to undefined behavior.
- Therefore, we explicitly delete the copy constructor and copy assignment operator to prevent accidental copies.
- The class still supports move semantics so that ownership can be transferred if needed.
- The class provides convenient methods: `record`, `synchronize`, and `elapsedTime` to interact with the event.
- In `main`, we demonstrate creating two events, recording a kernel, and measuring elapsed time between them.
- Error checking is performed via a helper macro `CUDA_CHECK` to simplify CUDA API call verification.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <chrono>

// Simple macro for CUDA error checking
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            std::cerr << "CUDA error in " << __FILE__ << ":"       \
                      << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error(cudaGetErrorString(err));     \
        }                                                          \
    } while (0)

// A trivial kernel for demonstration purposes
__global__ void dummyKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] * 2.0f;
}

// RAII wrapper around cudaEvent_t
class CudaEvent {
public:
    // Default constructor: create the event
    CudaEvent(cudaEventFlag_t flags = cudaEventDefault) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }

    // Destructor: destroy the event
    ~CudaEvent() {
        cudaEventDestroy(event_);
    }

    // Delete copy constructor
    CudaEvent(const CudaEvent&) = delete;
    // Delete copy assignment
    CudaEvent& operator=(const CudaEvent&) = delete;

    // Default move constructor
    CudaEvent(CudaEvent&& other) noexcept
        : event_(other.event_) {
        other.event_ = nullptr; // Prevent double destruction
    }

    // Default move assignment
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            cudaEventDestroy(event_); // destroy current
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    // Record the event on the given stream
    void record(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    // Wait until the event is completed
    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

    // Query if event is finished
    bool isFinished() const {
        cudaError_t err = cudaEventQuery(event_);
        return err == cudaSuccess;
    }

    // Compute elapsed time (in milliseconds) between this event and another event
    float elapsedTime(const CudaEvent& start) const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
        return ms;
    }

private:
    cudaEvent_t event_{nullptr};
};

int main() {
    try {
        const int N = 1 << 20; // 1M elements
        size_t size = N * sizeof(float);

        // Allocate device memory
        float* d_data = nullptr;
        CUDA_CHECK(cudaMalloc(&d_data, size));

        // Create a stream for the kernel
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Create two events to measure kernel execution
        CudaEvent startEvent;
        CudaEvent stopEvent;

        // Record start event
        startEvent.record(stream);

        // Launch dummy kernel
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        dummyKernel<<<blocks, threadsPerBlock, 0, stream>>>(d_data, N);
        CUDA_CHECK(cudaGetLastError());

        // Record stop event
        stopEvent.record(stream);

        // Wait for kernel to finish
        stopEvent.synchronize();

        // Measure elapsed time
        float ms = stopEvent.elapsedTime(startEvent);
        std::cout << "Kernel execution time: " << ms << " ms" << std::endl;

        // Clean up
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(d_data));

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
```