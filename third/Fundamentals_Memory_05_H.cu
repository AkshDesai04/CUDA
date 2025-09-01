/*
Aim of the program:
Write a simple C++ vector-like class for device memory. The constructor calls `cudaMalloc`, the destructor calls `cudaFree`.

Thought process:
- Need a C++ class template that stores a pointer to device memory and its size.
- Constructor should allocate memory on device using cudaMalloc.
- Destructor should free device memory using cudaFree.
- Provide minimal interface: size(), data() to get raw device pointer.
- Optionally implement a simple kernel to test functionality.
- Add error checking in constructor.
- Keep class lightweight, no copy constructor to avoid double free; maybe disable copy and provide move semantics.
- Include a small example in main to demonstrate usage.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

template <typename T>
class DeviceVector {
public:
    // Disable copy to avoid double free
    DeviceVector(const DeviceVector&) = delete;
    DeviceVector& operator=(const DeviceVector&) = delete;

    // Move semantics
    DeviceVector(DeviceVector&& other) noexcept
        : d_ptr_(other.d_ptr_), size_(other.size_) {
        other.d_ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceVector& operator=(DeviceVector&& other) noexcept {
        if (this != &other) {
            free();
            d_ptr_ = other.d_ptr_;
            size_ = other.size_;
            other.d_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Constructor: allocate device memory
    explicit DeviceVector(size_t n)
        : d_ptr_(nullptr), size_(n) {
        if (size_ == 0) return;
        cudaError_t err = cudaMalloc(&d_ptr_, size_ * sizeof(T));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
            d_ptr_ = nullptr;
            size_ = 0;
        }
    }

    // Destructor: free device memory
    ~DeviceVector() {
        free();
    }

    // Return raw device pointer
    __host__ T* data() { return d_ptr_; }
    __host__ const T* data() const { return d_ptr_; }

    // Return size
    __host__ size_t size() const { return size_; }

private:
    void free() {
        if (d_ptr_) {
            cudaFree(d_ptr_);
            d_ptr_ = nullptr;
        }
    }

    T* d_ptr_;
    size_t size_;
};

// Simple kernel to double each element
__global__ void double_elements(int* arr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] *= 2;
    }
}

int main() {
    const size_t N = 10;
    int h_data[N];
    for (size_t i = 0; i < N; ++i) h_data[i] = static_cast<int>(i);

    // Allocate device vector
    DeviceVector<int> d_vec(N);
    assert(d_vec.data() != nullptr);

    // Copy data from host to device
    cudaError_t err = cudaMemcpy(d_vec.data(), h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    double_elements<<<blocksPerGrid, threadsPerBlock>>>(d_vec.data(), N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_data, d_vec.data(), N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << stdendl;
        return 1;
    }

    // Print results
    std::cout << "Result after doubling on device:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
