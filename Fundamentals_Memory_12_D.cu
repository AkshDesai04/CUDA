/*
Add member functions `copy_to_device(const T* host_ptr, size_t count)` and `copy_to_host(T* host_ptr, size_t count)`.

My thinking: The task is to extend a CUDA C++ class that manages device memory to provide convenient copy operations. I will define a templated class DeviceArray that stores a pointer to device memory and the size. The new member functions will use cudaMemcpy to copy data between host and device. I will also add basic error checking and a simple kernel to demonstrate usage. The program will compile as a .cu file and can be built with nvcc.
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

// Simple macro for error checking
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " ";\
            std::cerr << cudaGetErrorString(err) << std::endl;                   \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// Templated device array class
template <typename T>
class DeviceArray {
public:
    DeviceArray(size_t n) : size_(n), d_ptr_(nullptr) {
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&d_ptr_, n * sizeof(T)));
        }
    }

    ~DeviceArray() {
        if (d_ptr_) {
            CUDA_CHECK(cudaFree(d_ptr_));
        }
    }

    // Copy data from host to device
    void copy_to_device(const T* host_ptr, size_t count) {
        if (!host_ptr) {
            std::cerr << "Null host pointer passed to copy_to_device." << std::endl;
            return;
        }
        if (count > size_) {
            std::cerr << "Count exceeds device array size in copy_to_device." << std::endl;
            return;
        }
        CUDA_CHECK(cudaMemcpy(d_ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Copy data from device to host
    void copy_to_host(T* host_ptr, size_t count) const {
        if (!host_ptr) {
            std::cerr << "Null host pointer passed to copy_to_host." << std::endl;
            return;
        }
        if (count > size_) {
            std::cerr << "Count exceeds device array size in copy_to_host." << std::endl;
            return;
        }
        CUDA_CHECK(cudaMemcpy(host_ptr, d_ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    // Accessor for the raw device pointer (used in kernels)
    T* data() { return d_ptr_; }
    const T* data() const { return d_ptr_; }

    size_t size() const { return size_; }

private:
    size_t size_;
    T* d_ptr_;
};

// Generic kernel to multiply each element by 2
template <typename T>
__global__ void multiply_by_two(T* d_data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] *= static_cast<T>(2);
    }
}

int main() {
    const size_t N = 10;
    std::vector<float> host_vec(N, 1.0f);          // Initialize with 1.0f
    std::vector<float> host_result(N, 0.0f);

    // Create device array
    DeviceArray<float> dev_arr(N);

    // Copy data to device
    dev_arr.copy_to_device(host_vec.data(), N);

    // Launch kernel
    const size_t threads_per_block = 256;
    const size_t blocks = (N + threads_per_block - 1) / threads_per_block;
    multiply_by_two<float><<<blocks, threads_per_block>>>(dev_arr.data(), N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    dev_arr.copy_to_host(host_result.data(), N);

    // Print results
    std::cout << "Results after kernel execution:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << "host_result[" << i << "] = " << host_result[i] << std::endl;
    }

    return 0;
}
