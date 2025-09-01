```cpp
/*
Aim of the program: Add member functions `copy_to_device(const T* host_ptr, size_t count)` and `copy_to_host(T* host_ptr, size_t count)`.

Thinking:
1. The goal is to create a CUDA C++ program that demonstrates how to copy data between host and device using a templated class.
2. I will implement a simple `DeviceArray<T>` class that manages a device pointer and its size.
3. The class will expose two member functions:
   - `copy_to_device(const T* host_ptr, size_t count)` to copy `count` elements from a host pointer to the device memory.
   - `copy_to_host(T* host_ptr, size_t count)` to copy `count` elements from the device memory back to a host pointer.
4. Error checking will be performed using CUDA runtime API functions and a helper macro.
5. In the `main` function, I will:
   - Create a host array of integers.
   - Allocate a `DeviceArray<int>` with the same size.
   - Copy data from the host to the device using `copy_to_device`.
   - Launch a simple kernel that increments each element on the device.
   - Copy the results back to the host using `copy_to_host`.
   - Print the results to verify correctness.
6. The entire code will be self-contained, compilable with `nvcc`, and will use only CUDA runtime API and standard C++ features.

*/

#include <iostream>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a launch or API call
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in file '" << __FILE__                    \
                      << "' in line " << __LINE__ << ": "                      \
                      << cudaGetErrorString(err) << std::endl;                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Templated class to manage a device array
template <typename T>
class DeviceArray {
public:
    DeviceArray(size_t sz) : size_(sz), d_ptr_(nullptr) {
        // Allocate device memory
        CHECK_CUDA(cudaMalloc(&d_ptr_, size_ * sizeof(T)));
    }

    ~DeviceArray() {
        if (d_ptr_) {
            cudaFree(d_ptr_);
        }
    }

    // Copy data from host to device
    void copy_to_device(const T* host_ptr, size_t count) {
        if (count > size_) {
            std::cerr << "Error: count > allocated size in copy_to_device." << std::endl;
            exit(EXIT_FAILURE);
        }
        CHECK_CUDA(cudaMemcpy(d_ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Copy data from device to host
    void copy_to_host(T* host_ptr, size_t count) const {
        if (count > size_) {
            std::cerr << "Error: count > allocated size in copy_to_host." << std::endl;
            exit(EXIT_FAILURE);
        }
        CHECK_CUDA(cudaMemcpy(host_ptr, d_ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    // Return raw device pointer (for kernel launches)
    T* data() { return d_ptr_; }
    const T* data() const { return d_ptr_; }

    size_t size() const { return size_; }

private:
    size_t size_;
    T* d_ptr_;
};

// Simple CUDA kernel that increments each element by 1
__global__ void increment_kernel(int* d_data, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_data[idx] += 1;
    }
}

int main() {
    const size_t N = 10;
    int h_array[N];
    for (size_t i = 0; i < N; ++i) {
        h_array[i] = static_cast<int>(i);
    }

    // Create device array
    DeviceArray<int> d_array(N);

    // Copy data from host to device
    d_array.copy_to_device(h_array, N);

    // Launch kernel to increment each element
    const size_t threadsPerBlock = 256;
    const size_t blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment_kernel<<<blocks, threadsPerBlock>>>(d_array.data(), N);
    CHECK_CUDA(cudaGetLastError()); // Check kernel launch

    // Copy results back to host
    int h_result[N];
    d_array.copy_to_host(h_result, N);

    // Print results
    std::cout << "Result after increment:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```