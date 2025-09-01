```cuda
/*
Wrap the memory allocation and deallocation in a C++ class using RAII (Constructor allocates, Destructor frees).

Thinking:
The goal is to create a simple CUDA program that demonstrates RAII for device memory.
- Create a template class `DeviceBuffer<T>` that manages a pointer `T* d_ptr` on the device.
- The constructor will take a size in elements and call `cudaMalloc`.
- The destructor will free the memory with `cudaFree`.
- Provide a `operator T*()` to get the raw device pointer for kernel launches.
- For safety, disable copy construction/assignment and provide move semantics.
- Write a trivial kernel that increments each element by 1.
- In `main`, allocate a `DeviceBuffer<int>` of a certain size, copy host data to device, launch kernel, copy back, and verify.
- Include basic error checking for CUDA calls.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

// Simple RAII wrapper for device memory
template<typename T>
class DeviceBuffer {
public:
    // Constructor allocates device memory of given number of elements
    explicit DeviceBuffer(size_t count) : size_(count), d_ptr_(nullptr) {
        if (count == 0) {
            throw std::runtime_error("DeviceBuffer: cannot allocate zero elements");
        }
        cudaError_t err = cudaMalloc(&d_ptr_, count * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
        }
    }

    // Destructor frees device memory
    ~DeviceBuffer() {
        if (d_ptr_) {
            cudaFree(d_ptr_);
        }
    }

    // Disable copy semantics
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Enable move semantics
    DeviceBuffer(DeviceBuffer&& other) noexcept : size_(other.size_), d_ptr_(other.d_ptr_) {
        other.d_ptr_ = nullptr;
        other.size_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (d_ptr_) cudaFree(d_ptr_);
            d_ptr_ = other.d_ptr_;
            size_ = other.size_;
            other.d_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Get raw device pointer
    T* get() const { return d_ptr_; }

    // Get size
    size_t size() const { return size_; }

private:
    size_t size_;
    T*    d_ptr_;
};

// Simple kernel that increments each element by 1
__global__ void incrementKernel(int* d_data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] += 1;
    }
}

// Utility to check CUDA errors
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " : " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const size_t N = 1024; // number of integers
    std::vector<int> h_vec(N);
    // Initialize host data
    for (size_t i = 0; i < N; ++i) h_vec[i] = static_cast<int>(i);

    // Allocate device buffer using RAII
    DeviceBuffer<int> d_buf(N);

    // Copy data from host to device
    checkCuda(cudaMemcpy(d_buf.get(), h_vec.data(), N * sizeof(int), cudaMemcpyHostToDevice), "Memcpy H2D");

    // Launch kernel
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    incrementKernel<<<blocks, threadsPerBlock>>>(d_buf.get(), N);
    checkCuda(cudaGetLastError(), "Kernel launch");
    checkCuda(cudaDeviceSynchronize(), "Kernel execution");

    // Copy result back to host
    checkCuda(cudaMemcpy(h_vec.data(), d_buf.get(), N * sizeof(int), cudaMemcpyDeviceToHost), "Memcpy D2H");

    // Verify results
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_vec[i] != static_cast<int>(i + 1)) {
            std::cerr << "Mismatch at index " << i << ": expected " << i+1 << ", got " << h_vec[i] << std::endl;
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Success: All values incremented correctly." << std::endl;
    } else {
        std::cout << "Failure: Data mismatch." << std::endl;
    }

    // DeviceBuffer destructor automatically frees memory

    return 0;
}
```