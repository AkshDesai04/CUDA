```cpp
/* 
Aim of the program:
Create a similar RAII wrapper class for CUDA events (`cudaEvent_t`).

Thinking:
The task is to provide a RAII (Resource Acquisition Is Initialization) style wrapper for CUDA events. CUDA events (`cudaEvent_t`) are opaque handles that require explicit creation (`cudaEventCreateWithFlags`) and destruction (`cudaEventDestroy`). In a C++ RAII class we want the constructor to create the event, and the destructor to destroy it automatically. We also need to prevent accidental copying (since copying would duplicate the handle without managing the underlying resource). We can, however, allow move semantics so that the wrapper can be transferred efficiently.

Functionalities to include:
1. Constructor that takes optional flags and creates the event.
2. Destructor that destroys the event.
3. Deleted copy constructor and copy assignment operator.
4. Move constructor and move assignment operator to transfer ownership.
5. Methods for common operations:
   - `record` to record the event on a stream.
   - `synchronize` to block until the event completes.
   - `query` to check completion status.
6. A getter for the raw `cudaEvent_t` handle if needed.
7. Basic error handling: if any CUDA call fails, throw a `std::runtime_error` with a descriptive message.

We'll also include a simple `main` that demonstrates creating an event, recording a trivial kernel, synchronizing, and printing a success message. This will ensure the wrapper works in a minimal example.

The code below compiles with a standard CUDA compiler (nvcc) and demonstrates the RAII wrapper usage.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>

class CudaEvent {
public:
    // Create an event with optional flags (default: cudaEventDefault)
    explicit CudaEvent(unsigned int flags = cudaEventDefault) {
        cudaError_t err = cudaEventCreateWithFlags(&event_, flags);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA event: " + std::string(cudaGetErrorString(err)));
        }
    }

    // Destructor: destroys the event
    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    // Delete copy constructor and copy assignment
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    // Move constructor
    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    // Move assignment operator
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            // Destroy current event
            if (event_ != nullptr) {
                cudaEventDestroy(event_);
            }
            // Transfer ownership
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    // Record the event on a given stream (default: stream 0)
    void record(cudaStream_t stream = 0) {
        cudaError_t err = cudaEventRecord(event_, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to record CUDA event: " + std::string(cudaGetErrorString(err)));
        }
    }

    // Block until the event completes
    void synchronize() {
        cudaError_t err = cudaEventSynchronize(event_);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to synchronize CUDA event: " + std::string(cudaGetErrorString(err)));
        }
    }

    // Query if the event has completed
    bool query() const {
        cudaError_t err = cudaEventQuery(event_);
        if (err == cudaSuccess) return true;
        if (err == cudaErrorNotReady) return false;
        throw std::runtime_error("Failed to query CUDA event: " + std::string(cudaGetErrorString(err)));
    }

    // Access the underlying cudaEvent_t handle
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_;
};

// A trivial CUDA kernel for demonstration purposes
__global__ void dummyKernel() {
    // No-op
}

int main() {
    try {
        // Ensure at least one CUDA device is available
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            throw std::runtime_error("No CUDA device available.");
        }

        // Create a CUDA event
        CudaEvent evt;

        // Launch a dummy kernel
        dummyKernel<<<1, 1>>>();

        // Record the event after kernel launch
        evt.record();

        // Wait for the event to complete
        evt.synchronize();

        // Check completion status
        if (evt.query()) {
            std::cout << "CUDA event completed successfully.\n";
        } else {
            std::cout << "CUDA event is not yet complete.\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
```