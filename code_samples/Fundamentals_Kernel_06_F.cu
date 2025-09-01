/*
Wrap the memory allocation and deallocation in a C++ class using RAII (Constructor allocates, Destructor frees).

The goal of this program is to demonstrate how to manage CUDA device memory safely in C++ using the RAII idiom.
In RAII, resource acquisition occurs during construction, and release happens automatically in the destructor,
so the user does not need to remember to call free functions explicitly.  The example below provides a
templated `DeviceArray` class that allocates a contiguous array on the GPU when constructed and frees it
when destroyed.  Copy semantics are deleted to avoid accidental double‑free, but move semantics are
implemented to allow transfer of ownership.

The main function exercises the class:
1. Creates a `DeviceArray<float>` of a given size.
2. Copies data from host to device.
3. Launches a simple kernel that multiplies each element by two.
4. Copies the results back to the host.
5. Prints the final array to verify correctness.

The program also includes basic CUDA error checking and a helper macro to keep the code concise.
*/

#include <iostream>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>

// Simple macro for checking CUDA errors
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(err) << std::endl;      \
            throw std::runtime_error("CUDA error");                         \
        }                                                                    \
    } while (0)

// RAII wrapper for CUDA device memory
template <typename T>
class DeviceArray {
public:
    // Allocate memory on construction
    explicit DeviceArray(size_t n) : size_(n), d_ptr_(nullptr) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMalloc(&d_ptr_, size_ * sizeof(T)));
        }
    }

    // Move constructor
    DeviceArray(DeviceArray&& other) noexcept
        : size_(other.size_), d_ptr_(other.d_ptr_) {
        other.d_ptr_ = nullptr;
        other.size_ = 0;
    }

    // Move assignment
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            free();  // Free current resources
            size_ = other.size_;
            d_ptr_ = other.d_ptr_;
            other.d_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Delete copy constructor and copy assignment to avoid double free
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    // Destructor frees memory
    ~DeviceArray() {
        free();
    }

    // Accessors
    T* get() { return d_ptr_; }
    const T* get() const { return d_ptr_; }
    size_t size() const { return size_; }

private:
    void free() {
        if (d_ptr_) {
            CUDA_CHECK(cudaFree(d_ptr_));
            d_ptr_ = nullptr;
        }
    }

    size_t size_;
    T* d_ptr_;
};

// Simple kernel that multiplies each element by two
__global__ void multiplyByTwo(float* d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] *= 2.0f;
    }
}

int main() {
    const int N = 10;
    std::vector<float> h_vec(N);

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_vec[i] = static_cast<float>(i + 1);
    }

    // Allocate device memory using RAII
    DeviceArray<float> d_vec(N);

    // Copy from host to device
    CUDA_CHECK(cudaMemcpy(d_vec.get(), h_vec.data(),
                          N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    multiplyByTwo<<<blocks, threadsPerBlock>>>(d_vec.get(), N);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_vec.data(), d_vec.get(),
                          N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "Result after kernel:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
