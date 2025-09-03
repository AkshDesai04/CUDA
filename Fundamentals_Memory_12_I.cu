```cpp
/*
Create a similar RAII wrapper class for CUDA events (`cudaEvent_t`).

Thought process:
- The goal is to provide an RAII (Resource Acquisition Is Initialization) wrapper that automatically manages the lifetime of a CUDA event. In CUDA, events are created with `cudaEventCreate` and destroyed with `cudaEventDestroy`. If a programmer forgets to destroy an event, it can lead to resource leaks, especially when many events are created dynamically.
- An RAII wrapper will:
  1. Create the event in its constructor.
  2. Destroy the event in its destructor.
  3. Disable copy semantics to avoid accidental double-destruction.
  4. Enable move semantics for efficient transfer of ownership.
- We should also provide convenient member functions to interact with the event:
  * `record(cudaStream_t stream = 0)` – records the event on a stream.
  * `synchronize()` – blocks until the event is complete.
  * `query()` – non-blocking check of event completion.
  * `elapsedTime(const CudaEvent& end, float* ms)` – computes time between two events.
- For error handling, a small helper `checkCuda` function will throw a `std::runtime_error` when CUDA API calls fail. This keeps the wrapper code clean and avoids repetitive error checks.
- The code will also include a small example in `main` that demonstrates creating two events, recording them around a simple kernel launch, synchronizing, and printing the elapsed time.
- The implementation will be self-contained in a `.cu` file. It will include necessary headers (`<cuda_runtime.h>`, `<iostream>`, `<stdexcept>`), the RAII wrapper class `CudaEvent`, and a `main` function for demonstration.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

/* Helper to check CUDA errors */
inline void checkCuda(cudaError_t err, const char* msg = nullptr) {
    if (err != cudaSuccess) {
        std::string errorMsg = msg ? std::string(msg) : "CUDA error: ";
        errorMsg += cudaGetErrorString(err);
        throw std::runtime_error(errorMsg);
    }
}

/* RAII wrapper for cudaEvent_t */
class CudaEvent {
public:
    /* Default constructor creates a CUDA event with default flags */
    CudaEvent(unsigned int flags = cudaEventDefault) {
        cudaError_t err = cudaEventCreateWithFlags(&event_, flags);
        checkCuda(err, "Failed to create CUDA event");
    }

    /* Destructor destroys the CUDA event */
    ~CudaEvent() {
        if (valid_) {
            cudaEventDestroy(event_);
            valid_ = false;
        }
    }

    /* Move constructor */
    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_), valid_(other.valid_) {
        other.valid_ = false;
    }

    /* Move assignment */
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (valid_) {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            valid_ = other.valid_;
            other.valid_ = false;
        }
        return *this;
    }

    /* Delete copy constructor and copy assignment */
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    /* Record the event on a given stream (default stream 0) */
    void record(cudaStream_t stream = 0) {
        checkCuda(cudaEventRecord(event_, stream), "Failed to record CUDA event");
    }

    /* Block until the event completes */
    void synchronize() const {
        checkCuda(cudaEventSynchronize(event_), "Failed to synchronize CUDA event");
    }

    /* Query the status of the event (non-blocking) */
    bool query() const {
        cudaError_t err = cudaEventQuery(event_);
        if (err == cudaSuccess) return true;
        if (err == cudaErrorNotReady) return false;
        checkCuda(err, "Failed to query CUDA event");
        return false; // Unreachable, but keeps compiler happy
    }

    /* Compute elapsed time in milliseconds between this event and another */
    void elapsedTime(const CudaEvent& end, float* ms) const {
        checkCuda(cudaEventElapsedTime(ms, event_, end.event_), "Failed to compute elapsed time");
    }

    /* Access the underlying cudaEvent_t (for advanced usage) */
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_;
    bool valid_ = true;
};

/* Simple kernel that does nothing but delays for a few milliseconds */
__global__ void dummyKernel(int n) {
    // Intentionally left empty; just for timing purposes
    for (int i = 0; i < n; ++i) {
        // Prevent compiler from optimizing away
        __syncthreads();
    }
}

int main() {
    try {
        // Create two events: start and stop
        CudaEvent startEvent;
        CudaEvent stopEvent;

        // Launch a dummy kernel on the default stream
        int N = 1 << 20; // Number of iterations
        dummyKernel<<<1, 1>>>(N);
        checkCuda(cudaGetLastError(), "Kernel launch failed");

        // Record events
        startEvent.record();
        stopEvent.record();

        // Wait for both events to complete
        startEvent.synchronize();
        stopEvent.synchronize();

        // Compute elapsed time
        float ms = 0.0f;
        startEvent.elapsedTime(stopEvent, &ms);

        std::cout << "Elapsed time: " << ms << " ms" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```