/*
Write a simple C++ vector-like class for device memory. The constructor calls cudaMalloc, the destructor calls cudaFree.
 
In order to provide a minimal yet functional C++ wrapper around CUDA device memory, I chose to implement a
template class `DeviceVector<T>`.  The template allows the user to store any trivially copyable type (int,
float, double, structs, etc.).  The class manages a raw pointer `T* d_data_` that is allocated with
`cudaMalloc` in the constructor and freed with `cudaFree` in the destructor.  Basic utilities such as
`size()`, `data()` and an overloaded `operator[]` (which simply returns a device pointer offset) are
provided.  I also included a small helper function `copyFromHost` and `copyToHost` to demonstrate how a
user can transfer data between host and device, and a trivial CUDA kernel `incrementKernel` to
illustrate usage.  This design keeps the code compact, self‑contained, and suitable for quick
experimentation or educational purposes.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

// Utility macro for error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " code=" << err << " \"" << cudaGetErrorString(err) \
                      << "\"\n";                                            \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// Simple vector-like class for device memory
template <typename T>
class DeviceVector {
public:
    // Constructor: allocate device memory
    explicit DeviceVector(size_t n)
        : size_(n), d_data_(nullptr) {
        CUDA_CHECK(cudaMalloc(&d_data_, size_ * sizeof(T)));
    }

    // Destructor: free device memory
    ~DeviceVector() {
        if (d_data_) {
            CUDA_CHECK(cudaFree(d_data_));
        }
    }

    // Disable copy semantics
    DeviceVector(const DeviceVector&) = delete;
    DeviceVector& operator=(const DeviceVector&) = delete;

    // Enable move semantics
    DeviceVector(DeviceVector&& other) noexcept
        : size_(other.size_), d_data_(other.d_data_) {
        other.size_ = 0;
        other.d_data_ = nullptr;
    }

    DeviceVector& operator=(DeviceVector&& other) noexcept {
        if (this != &other) {
            // Free current resources
            if (d_data_) {
                CUDA_CHECK(cudaFree(d_data_));
            }
            // Steal resources
            size_ = other.size_;
            d_data_ = other.d_data_;
            other.size_ = 0;
            other.d_data_ = nullptr;
        }
        return *this;
    }

    // Return size of the vector
    size_t size() const { return size_; }

    // Return raw device pointer
    T* data() const { return d_data_; }

    // Simple device pointer offset (no bounds checking)
    __device__ T& operator[](size_t idx) const { return d_data_[idx]; }

    // Host-side copy from host array
    void copyFromHost(const T* h_data, size_t count) {
        assert(count <= size_);
        CUDA_CHECK(cudaMemcpy(d_data_, h_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Host-side copy to host array
    void copyToHost(T* h_data, size_t count) const {
        assert(count <= size_);
        CUDA_CHECK(cudaMemcpy(h_data, d_data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    size_t size_;
    T*    d_data_;
};

// Sample kernel that increments each element
template <typename T>
__global__ void incrementKernel(T* data, size_t n, T inc) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += inc;
    }
}

int main() {
    const size_t N = 10;
    DeviceVector<int> devVec(N);

    // Initialize host data
    int h_vec[N];
    for (size_t i = 0; i < N; ++i) {
        h_vec[i] = static_cast<int>(i);
    }

    // Copy to device
    devVec.copyFromHost(h_vec, N);

    // Launch kernel to add 5 to each element
    const int inc = 5;
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    incrementKernel<int><<<blocks, threadsPerBlock>>>(devVec.data(), N, inc);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back to host
    int h_result[N];
    devVec.copyToHost(h_result, N);

    // Print result
    std::cout << "Result after incrementing by " << inc << ":\n";
    for (size_t i = 0; i < N; ++i) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
