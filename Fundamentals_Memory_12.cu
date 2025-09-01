```cpp
/* 
Aim of the program (as passed in the prompt word for word): 
Write a C++ wrapper class for device memory. The constructor should accept a size, call `cudaMalloc`, and store the pointer. The destructor should call `cudaFree`. This is a basic implementation of the RAII (Resource Acquisition Is Initialization) principle for CUDA pointers.

Thinking process:
1. The goal is to provide a RAII wrapper that manages device memory automatically, ensuring that memory is freed when the object goes out of scope.
2. The wrapper will store a `void*` pointer to the device memory and the requested size (for potential debugging or future extensions).
3. The constructor takes the size in bytes, calls `cudaMalloc`, and checks the return status. If allocation fails, it throws a C++ exception (std::runtime_error) with an informative message.
4. The destructor calls `cudaFree` on the stored pointer, but only if it is non‑null.
5. Copy semantics are disabled to avoid double free. Move semantics are provided to allow ownership transfer.
6. For demonstration, a simple `main` function allocates a buffer, prints the device pointer, and then exits, causing the destructor to run.
7. The file is a .cu file, suitable for compilation with `nvcc`. All required headers (`cuda_runtime.h`, `iostream`, `stdexcept`) are included.
8. Error handling is minimal but sufficient for illustration; real code might use more sophisticated error handling or a helper macro.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

/**
 * @brief RAII wrapper for CUDA device memory.
 *
 * This class allocates device memory in its constructor and frees it in its destructor,
 * ensuring that memory is properly released even if exceptions are thrown.
 */
class DeviceMemory {
public:
    // Disable default constructor
    DeviceMemory() = delete;

    // Constructor: allocate `size` bytes on the device
    explicit DeviceMemory(size_t size) : ptr_(nullptr), size_(size) {
        cudaError_t err = cudaMalloc(&ptr_, size_);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        }
    }

    // Destructor: free the allocated device memory
    ~DeviceMemory() {
        if (ptr_) {
            cudaError_t err = cudaFree(ptr_);
            if (err != cudaSuccess) {
                // In a destructor, throwing is dangerous; log the error instead
                std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }

    // Delete copy constructor and copy assignment to avoid double free
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    // Move constructor
    DeviceMemory(DeviceMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    // Move assignment
    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            // Free current memory
            if (ptr_) {
                cudaFree(ptr_);
            }
            // Steal resources
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Accessors
    void* get() const { return ptr_; }
    size_t size() const { return size_; }

private:
    void* ptr_;
    size_t size_;
};

int main() {
    try {
        const size_t bytes = 1024; // Allocate 1 KB on the device
        DeviceMemory devMem(bytes);
        std::cout << "Allocated " << bytes << " bytes on device at address "
                  << devMem.get() << std::endl;

        // When `devMem` goes out of scope, its destructor will automatically free the memory.
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
```