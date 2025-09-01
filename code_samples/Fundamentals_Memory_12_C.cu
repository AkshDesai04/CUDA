/*
Aim of the program:
Add a member function `get()` that returns the raw device pointer.

Thinking:
- The user requests a member function that returns the raw device pointer. 
- To demonstrate this, I design a simple CUDA wrapper class `DeviceArray<T>` that manages a contiguous block of memory on the GPU.
- The class will:
  - Allocate device memory in the constructor.
  - Free it in the destructor.
  - Provide a `get()` method that returns the raw device pointer.
  - Provide a `size()` method to know the number of elements.
  - Provide `copyFromHost` and `copyToHost` utilities to move data.
- I will also include a simple kernel that increments each element by 1, using the raw pointer obtained via `get()`.
- In `main()`, I'll allocate a small array, initialize host data, copy to device, launch the kernel, copy back, and print the results.
- All CUDA error checking is handled by a simple macro `CUDA_CHECK`.
- The code is self-contained, can compile as a `.cu` file, and demonstrates the requested `get()` functionality.
*/

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " code=" << err << " \"" << cudaGetErrorString(err)\
                      << "\"\n";                                            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Simple CUDA kernel that adds 1 to each element of an array
template <typename T>
__global__ void incrementKernel(T* d_arr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += static_cast<T>(1);
    }
}

// Device array wrapper class
template <typename T>
class DeviceArray {
public:
    DeviceArray(size_t n) : size_(n), d_ptr_(nullptr) {
        CUDA_CHECK(cudaMalloc(&d_ptr_, size_ * sizeof(T)));
    }

    ~DeviceArray() {
        if (d_ptr_) {
            CUDA_CHECK(cudaFree(d_ptr_));
        }
    }

    // Delete copy constructor and assignment to avoid accidental copies
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    // Move constructor
    DeviceArray(DeviceArray&& other) noexcept
        : size_(other.size_), d_ptr_(other.d_ptr_) {
        other.d_ptr_ = nullptr;
        other.size_ = 0;
    }

    // Move assignment
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            if (d_ptr_) {
                CUDA_CHECK(cudaFree(d_ptr_));
            }
            d_ptr_ = other.d_ptr_;
            size_ = other.size_;
            other.d_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Return raw device pointer
    T* get() const { return d_ptr_; }

    size_t size() const { return size_; }

    // Copy data from host to device
    void copyFromHost(const T* h_src) {
        CUDA_CHECK(cudaMemcpy(d_ptr_, h_src, size_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Copy data from device to host
    void copyToHost(T* h_dst) const {
        CUDA_CHECK(cudaMemcpy(h_dst, d_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    size_t size_;
    T* d_ptr_;
};

int main() {
    const size_t N = 10;
    int h_data[N];
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = static_cast<int>(i);
    }

    // Create device array
    DeviceArray<int> d_arr(N);
    d_arr.copyFromHost(h_data);

    // Launch kernel to increment each element
    const size_t threadsPerBlock = 256;
    const size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    incrementKernel<int><<<blocksPerGrid, threadsPerBlock>>>(d_arr.get(), d_arr.size());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    int h_result[N];
    d_arr.copyToHost(h_result);

    // Print results
    std::cout << "Result after increment:\n";
    for (size_t i = 0; i < N; ++i) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
