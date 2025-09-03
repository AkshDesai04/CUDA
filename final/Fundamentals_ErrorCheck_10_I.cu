```cuda
/*
Aim of the program: Make the class non-copyable by deleting the copy constructor and copy assignment operator, as `cudaEvent_t` handles cannot be trivially copied.

Thinking:  
- We want a RAII wrapper around a `cudaEvent_t` so that the event is created in the constructor and destroyed in the destructor.  
- Because CUDA events are opaque handles that cannot be copied safely, the wrapper must be non‑copyable.  
- We provide a move constructor and move assignment so that ownership can be transferred without copying.  
- The class offers basic operations (`record`, `synchronize`, `query`) that forward to the CUDA runtime.  
- In `main` we create two events, record on one, and synchronize on the other to demonstrate usage.  
- Simple error checking is added to make the example robust.  
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

/* Utility macro for CUDA error checking */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__      \
                      << " : " << cudaGetErrorString(err) << std::endl;       \
            throw std::runtime_error(cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

/* RAII wrapper for cudaEvent_t */
class CudaEvent {
public:
    /* Default constructor creates an event */
    CudaEvent(cudaEventFlags flags = cudaEventDefault) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }

    /* Move constructor */
    CudaEvent(CudaEvent&& other) noexcept
        : event_(other.event_) {
        other.event_ = 0;  // Invalidate the moved-from handle
    }

    /* Move assignment */
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            destroy();
            event_ = other.event_;
            other.event_ = 0;
        }
        return *this;
    }

    /* Delete copy constructor */
    CudaEvent(const CudaEvent&) = delete;
    /* Delete copy assignment */
    CudaEvent& operator=(const CudaEvent&) = delete;

    /* Destructor destroys the event */
    ~CudaEvent() {
        destroy();
    }

    /* Record an event on a stream */
    void record(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    /* Wait for the event to complete */
    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

    /* Query event status without blocking */
    bool query() const {
        cudaError_t err = cudaEventQuery(event_);
        if (err == cudaSuccess) {
            return true;
        } else if (err == cudaErrorNotReady) {
            return false;
        } else {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }

    /* Get underlying CUDA event */
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = 0;

    void destroy() {
        if (event_ != 0) {
            cudaEventDestroy(event_);
            event_ = 0;
        }
    }
};

/* Simple kernel to demonstrate usage */
__global__ void dummyKernel(int *data, int val, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = val;
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    int *d_data = nullptr;

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));

    /* Create CUDA events */
    CudaEvent startEvent;
    CudaEvent stopEvent;

    /* Launch kernel and record events */
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    startEvent.record(); // Record before kernel launch

    dummyKernel<<<grid, block>>>(d_data, 42, N);

    stopEvent.record(); // Record after kernel launch

    /* Wait for kernel to finish */
    stopEvent.synchronize();

    /* Optional: verify results */
    int *h_data = new int[N];
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_data[i] != 42) {
            ok = false;
            std::cerr << "Mismatch at index " << i << ": " << h_data[i] << std::endl;
            break;
        }
    }
    std::cout << "Verification " << (ok ? "succeeded" : "failed") << std::endl;

    /* Clean up */
    delete[] h_data;
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
```