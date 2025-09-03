```cpp
/*
Write a C++ wrapper class for device memory. The constructor should accept a size, call `cudaMalloc`, and store the pointer. The destructor should call `cudaFree`. This is a basic implementation of the RAII (Resource Acquisition Is Initialization) principle for CUDA pointers.

Thought process:
- Need a class that manages a CUDA device pointer.
- Constructor takes a size (in bytes), calls cudaMalloc, stores pointer and size.
- Destructor calls cudaFree if pointer is not null.
- Prevent copy semantics (copy constructor and copy assignment deleted).
- Provide move semantics for convenience.
- Provide accessor methods to get raw pointer and size.
- Use error handling: if cudaMalloc fails, throw std::runtime_error.
- Include necessary headers: <cuda_runtime.h>, <stdexcept>.
- Keep implementation in a .cu file; no external dependencies.
- Add a small test in main to demonstrate usage (optional but helpful for compilation).
*/

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <cstdio>

// C++ wrapper for CUDA device memory using RAII
class CudaDeviceBuffer
{
public:
    // Constructor: allocate memory of given size (in bytes)
    explicit CudaDeviceBuffer(std::size_t size)
        : ptr_(nullptr), size_(size)
    {
        if (size_ == 0) {
            // Zero-sized allocation is a no-op; keep ptr_ nullptr
            return;
        }
        cudaError_t err = cudaMalloc(&ptr_, size_);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        }
    }

    // Destructor: free device memory
    ~CudaDeviceBuffer()
    {
        if (ptr_) {
            cudaError_t err = cudaFree(ptr_);
            if (err != cudaSuccess) {
                // In destructor, we cannot throw. Log the error.
                std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }

    // Delete copy constructor and copy assignment
    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    // Move constructor
    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    // Move assignment
    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept
    {
        if (this != &other) {
            // Free existing resource
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

    // Accessors
    void* data() noexcept { return ptr_; }
    const void* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return size_; }

private:
    void* ptr_;
    std::size_t size_;
};

// Example usage: allocate 1024 bytes on device
int main()
{
    try {
        CudaDeviceBuffer buf(1024);
        std::cout << "Allocated " << buf.size() << " bytes on device at " << buf.data() << std::endl;
        // Use the buffer as needed...
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
```