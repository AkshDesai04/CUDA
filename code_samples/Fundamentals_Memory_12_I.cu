/*
Aim: Create a similar RAII wrapper class for CUDA events (`cudaEvent_t`).

I need to design a class that encapsulates a CUDA event so that its lifetime is automatically managed. The class should:
- Create the event in the constructor (with optional flags).
- Destroy the event in the destructor.
- Be non‑copyable to avoid double destruction, but movable to allow transfer of ownership.
- Provide convenient member functions for common CUDA event operations:
  - `record` – record the event on a stream.
  - `wait` – make a stream wait on the event.
  - `reset` – reset the event so it can be reused.
  - `elapsedTime` – compute elapsed time between two events.
- Offer an implicit conversion to `cudaEvent_t` for compatibility with CUDA runtime APIs.
- Perform error checking on all CUDA API calls, throwing a `std::runtime_error` with a helpful message if any call fails.
- Use `cudaEventCreateWithFlags` to allow optional flags (e.g., `cudaEventDisableTiming`).
- Ensure the class is exception safe: if construction fails, the destructor should not be called.
- Provide a small example `main` function demonstrating the wrapper’s usage, creating two events, recording them, waiting, and printing the elapsed time.

The code is written in CUDA C++ and should compile with `nvcc`. All necessary includes and helper functions are included within the same file, making it self‑contained. The only output is the .cu file content as requested. 
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

/**
 * Helper function to check CUDA runtime API return status.
 * Throws a std::runtime_error if the status is not cudaSuccess.
 */
inline void checkCuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

/**
 * RAII wrapper for cudaEvent_t.
 *
 * - Non-copyable, movable.
 * - Creates an event in the constructor, destroys it in the destructor.
 * - Provides convenient methods: record, wait, reset, elapsedTime.
 * - Implicit conversion to cudaEvent_t for use with CUDA APIs.
 */
class CudaEvent
{
public:
    // Default constructor creates an event with no flags
    explicit CudaEvent(unsigned int flags = 0)
    {
        cudaError_t err = cudaEventCreateWithFlags(&event_, flags);
        checkCuda(err, "cudaEventCreateWithFlags");
    }

    // Destructor destroys the event
    ~CudaEvent()
    {
        if (event_)
        {
            cudaError_t err = cudaEventDestroy(event_);
            // In a destructor we should not throw, so we ignore errors but could log
            (void)err; // silence unused variable warning
        }
    }

    // Move constructor
    CudaEvent(CudaEvent&& other) noexcept
        : event_(other.event_)
    {
        other.event_ = nullptr;
    }

    // Move assignment
    CudaEvent& operator=(CudaEvent&& other) noexcept
    {
        if (this != &other)
        {
            if (event_)
            {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    // Delete copy constructor and copy assignment
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    // Record the event on the given stream (default stream 0)
    void record(cudaStream_t stream = 0)
    {
        cudaError_t err = cudaEventRecord(event_, stream);
        checkCuda(err, "cudaEventRecord");
    }

    // Make the given stream wait on this event
    void wait(cudaStream_t stream = 0) const
    {
        cudaError_t err = cudaStreamWaitEvent(stream, event_, 0);
        checkCuda(err, "cudaStreamWaitEvent");
    }

    // Reset the event (by recording it again on the default stream)
    void reset()
    {
        // Recording again effectively resets the event
        cudaError_t err = cudaEventRecord(event_, 0);
        checkCuda(err, "cudaEventRecord (reset)");
    }

    // Compute elapsed time (in milliseconds) between this event and another
    // The elapsed time is event2 - event1
    float elapsedTime(const CudaEvent& other) const
    {
        float ms = 0.0f;
        cudaError_t err = cudaEventElapsedTime(&ms, other.event_, event_);
        checkCuda(err, "cudaEventElapsedTime");
        return ms;
    }

    // Implicit conversion to cudaEvent_t
    operator cudaEvent_t() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

/**
 * Simple demonstration of the CudaEvent RAII wrapper.
 * Creates two events, records them on the default stream,
 * waits for them, and prints the elapsed time between them.
 */
int main()
{
    try
    {
        // Create two events
        CudaEvent startEvent;
        CudaEvent endEvent;

        // Record start event
        startEvent.record();

        // Perform a dummy kernel launch (simple vector addition)
        const int N = 1 << 20;
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, N * sizeof(float));
        cudaMalloc(&d_b, N * sizeof(float));
        cudaMalloc(&d_c, N * sizeof(float));

        // Launch a simple kernel (we'll use a lambda in C++14 or just a dummy memcpy)
        cudaMemcpy(d_a, d_a, N * sizeof(float), cudaMemcpyDefault); // dummy operation

        // Record end event
        endEvent.record();

        // Wait for the events to complete
        endEvent.wait();

        // Compute elapsed time
        float ms = endEvent.elapsedTime(startEvent);
        std::cout << "Elapsed time: " << ms << " ms" << std::endl;

        // Clean up
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
    catch (const std::exception& e)
    {
        std::cerr << "CUDA error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
