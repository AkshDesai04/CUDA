```cpp
/*
Show how RAII (e.g., your device memory wrapper class) works well with exceptions to ensure cudaFree is called even if an error occurs mid-function.

I started by thinking about how exception safety in C++ is typically achieved using RAII: objects that own resources are cleaned up in their destructors. For CUDA device memory, the resource is the memory allocated with cudaMalloc, which must be released with cudaFree. If an exception is thrown after allocation, normal control flow is interrupted and stack unwinding calls the destructors of any local objects that have already been constructed. By wrapping the device pointer in a small class whose destructor calls cudaFree, we guarantee that the memory is freed even when an exception propagates out of a function.

The program below demonstrates this idea. The `DeviceBuffer` class allocates memory in its constructor and frees it in its destructor. In `doWork` we allocate a buffer and then deliberately throw an exception. In `main` we catch that exception and check that the memory has been freed by querying the free GPU memory before and after the exception. The destructor prints a message so we can see that it is executed during stack unwinding.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// RAII wrapper for device memory
class DeviceBuffer {
public:
    // Allocate `size` bytes on the device
    DeviceBuffer(size_t size) : ptr_(nullptr), size_(size) {
        cudaError_t err = cudaMalloc(&ptr_, size_);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }

    // Disable copying
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Allow moving
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            // Free current memory
            if (ptr_) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Destructor frees the device memory
    ~DeviceBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
            std::cout << "DeviceBuffer destructor called: freed "
                      << size_ << " bytes\n";
        }
    }

    // Access the raw pointer
    void* ptr() const { return ptr_; }

private:
    void* ptr_;
    size_t size_;
};

// Function that intentionally throws an exception after allocating device memory
void doWork() {
    // Allocate a buffer of 5 integers
    DeviceBuffer<int> buf(5 * sizeof(int));

    // Simulate some work...
    std::cout << "doWork: allocated buffer, now throwing exception\n";

    // Throw an exception to trigger stack unwinding
    throw std::runtime_error("Simulated error in doWork");
}

int main() {
    size_t freeBefore, totalBefore;
    cudaMemGetInfo(&freeBefore, &totalBefore);
    std::cout << "GPU free memory before: " << freeBefore << " bytes\n";

    try {
        doWork();
    } catch (const std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << '\n';
    }

    size_t freeAfter, totalAfter;
    cudaMemGetInfo(&freeAfter, &totalAfter);
    std::cout << "GPU free memory after: " << freeAfter << " bytes\n";

    // Verify that memory was indeed freed
    if (freeAfter > freeBefore) {
        std::cout << "Memory was successfully freed by RAII.\n";
    } else {
        std::cout << "Memory was NOT freed properly.\n";
    }

    return 0;
}
```