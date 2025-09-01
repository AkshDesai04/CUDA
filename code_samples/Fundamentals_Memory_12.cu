```cpp
/*
Write a C++ wrapper class for device memory. The constructor should accept a size, call `cudaMalloc`, and store the pointer. The destructor should call `cudaFree`. This is a basic implementation of the RAII (Resource Acquisition Is Initialization) principle for CUDA pointers.

Thinking:
1. The goal is to manage CUDA device memory safely using RAII. 
2. The constructor will allocate memory with cudaMalloc and store the returned pointer and the requested size. 
3. The destructor will free the allocated memory with cudaFree if the pointer is non‑null. 
4. To prevent accidental copying (which would lead to double‑free), copy constructor and copy assignment are deleted. 
5. Move semantics are provided so the resource can be transferred safely. 
6. Basic error handling is added: if cudaMalloc fails, throw a std::runtime_error with the CUDA error string. 
7. The class exposes a `get()` accessor to retrieve the raw device pointer and a `size()` accessor. 
8. No additional features (like overloading operators) are included to keep the implementation minimal. 
9. The code includes the necessary CUDA runtime header and standard headers for exceptions.
*/

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

class DeviceMemory {
private:
    void* ptr_;
    size_t size_;

public:
    // Constructor: allocate device memory of given size
    explicit DeviceMemory(size_t size) : ptr_(nullptr), size_(size) {
        cudaError_t err = cudaMalloc(&ptr_, size_);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        }
    }

    // Destructor: free device memory
    ~DeviceMemory() {
        if (ptr_) {
            cudaError_t err = cudaFree(ptr_);
            if (err != cudaSuccess) {
                // In a destructor, we cannot throw; log the error
                std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }

    // Disable copy semantics
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    // Enable move semantics
    DeviceMemory(DeviceMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            // Free existing resource
            if (ptr_) {
                cudaFree(ptr_);
            }
            // Move resource
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
};

// Example usage (optional)
/*
int main() {
    try {
        DeviceMemory devMem(1024 * 1024); // 1 MB
        std::cout << "Allocated " << devMem.size() << " bytes on device." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
*/
```