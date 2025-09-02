```cpp
/*
Add a move constructor and a move assignment operator to allow for efficient transfers of ownership of the device pointer.

Thought process:
1. We need a CUDA-enabled C++ program (.cu file) that defines a class managing a device pointer.
2. The class should allocate memory on the GPU in its constructor and free it in its destructor.
3. To prevent accidental copying, we delete the copy constructor and copy assignment operator.
4. Implement a move constructor that transfers the device pointer and size from the source object to the new object, leaving the source in a valid but empty state (pointer = nullptr, size = 0).
5. Implement a move assignment operator that first frees any existing device memory in the destination, then steals the pointer and size from the source, and finally resets the source.
6. Provide a simple main function to demonstrate allocation, move construction, and move assignment, printing diagnostics to confirm correct behavior.
7. Use CUDA runtime API functions `cudaMalloc`, `cudaFree`, and basic error checking macros for simplicity.
8. Ensure all code is contained in one .cu file, with no external dependencies beyond the CUDA runtime and standard headers.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <utility> // for std::move

// Simple error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__      \
                      << " - " << cudaGetErrorString(err) << std::endl;       \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// Class that manages a device pointer
class GPUBuffer {
public:
    // Constructor: allocate device memory
    explicit GPUBuffer(size_t size)
        : d_ptr_(nullptr), size_(size) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMalloc(&d_ptr_, size_ * sizeof(float)));
            std::cout << "GPUBuffer: Allocated " << size_ << " floats at "
                      << static_cast<void*>(d_ptr_) << std::endl;
        }
    }

    // Destructor: free device memory
    ~GPUBuffer() {
        if (d_ptr_) {
            CUDA_CHECK(cudaFree(d_ptr_));
            std::cout << "GPUBuffer: Freed memory at "
                      << static_cast<void*>(d_ptr_) << std::endl;
        }
    }

    // Delete copy constructor and copy assignment to avoid accidental copies
    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;

    // Move constructor: transfer ownership
    GPUBuffer(GPUBuffer&& other) noexcept
        : d_ptr_(other.d_ptr_), size_(other.size_) {
        other.d_ptr_ = nullptr;
        other.size_  = 0;
        std::cout << "GPUBuffer: Moved constructor used. New ptr: "
                  << static_cast<void*>(d_ptr_)
                  << ", Source ptr reset to nullptr." << std::endl;
    }

    // Move assignment operator: free existing memory, then transfer ownership
    GPUBuffer& operator=(GPUBuffer&& other) noexcept {
        if (this != &other) {
            // Free current resource
            if (d_ptr_) {
                CUDA_CHECK(cudaFree(d_ptr_));
                std::cout << "GPUBuffer: Freed memory at "
                          << static_cast<void*>(d_ptr_) << " during move assignment." << std::endl;
            }
            // Transfer ownership
            d_ptr_ = other.d_ptr_;
            size_  = other.size_;
            // Reset source
            other.d_ptr_ = nullptr;
            other.size_  = 0;
            std::cout << "GPUBuffer: Move assignment used. New ptr: "
                      << static_cast<void*>(d_ptr_)
                      << ", Source ptr reset to nullptr." << std::endl;
        }
        return *this;
    }

    // Accessors
    float* data() const { return d_ptr_; }
    size_t size() const { return size_; }

private:
    float* d_ptr_;
    size_t size_;
};

int main() {
    std::cout << "=== Move Constructor Demo ===" << std::endl;
    GPUBuffer buf1(10); // Allocate buffer of 10 floats
    // Move construct buf2 from buf1
    GPUBuffer buf2(std::move(buf1));
    std::cout << "After move construction:" << std::endl;
    std::cout << "buf1 ptr: " << static_cast<void*>(buf1.data())
              << ", size: " << buf1.size() << std::endl;
    std::cout << "buf2 ptr: " << static_cast<void*>(buf2.data())
              << ", size: " << buf2.size() << std::endl;

    std::cout << "\n=== Move Assignment Demo ===" << std::endl;
    GPUBuffer buf3(20); // Allocate buffer of 20 floats
    std::cout << "Before move assignment:" << std::endl;
    std::cout << "buf2 ptr: " << static_cast<void*>(buf2.data())
              << ", size: " << buf2.size() << std::endl;
    std::cout << "buf3 ptr: " << static_cast<void*>(buf3.data())
              << ", size: " << buf3.size() << std::endl;

    // Move assign buf3 to buf2
    buf2 = std::move(buf3);
    std::cout << "After move assignment:" << std::endl;
    std::cout << "buf2 ptr: " << static_cast<void*>(buf2.data())
              << ", size: " << buf2.size() << std::endl;
    std::cout << "buf3 ptr: " << static_cast<void*>(buf3.data())
              << ", size: " << buf3.size() << std::endl;

    return 0;
}
```