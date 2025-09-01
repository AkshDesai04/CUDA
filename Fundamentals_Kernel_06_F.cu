```cpp
/*
Wrap the memory allocation and deallocation in a C++ class using RAII (Constructor allocates, Destructor frees).

Thinking:
- The goal is to demonstrate how to encapsulate CUDA device memory management inside a C++ RAII class.
- The class will allocate memory in its constructor and free it in its destructor, ensuring exception safety.
- We'll implement a template class `DeviceBuffer<T>` that stores a device pointer and the number of elements.
- The class will provide:
  * A constructor taking the number of elements (or size in bytes) and allocating device memory.
  * A destructor that frees the allocated memory.
  * A method to get the raw device pointer (`data()`).
  * A method to get the size (`size()`).
  * Copy constructors and assignment operators are deleted to avoid accidental double frees.
  * Optionally, we can provide a `copy_from_host` method to copy data from host to device.
- In the `main` function, we'll:
  * Create a host vector of floats.
  * Use `DeviceBuffer<float>` to allocate device memory.
  * Copy the host data to the device.
  * Launch a simple kernel that multiplies each element by 2.
  * Copy the results back to host and print a few elements.
- The kernel will be very small: `__global__ void multiply_by_two(float* d, int N)`.
- This example will compile with `nvcc` and run on a CUDA-capable device.
- No external dependencies beyond the standard library and CUDA runtime.

The code below follows these design choices and demonstrates RAII for CUDA memory.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>

/* RAII wrapper for CUDA device memory */
template <typename T>
class DeviceBuffer {
public:
    // Constructor: allocate device memory for `count` elements
    explicit DeviceBuffer(size_t count) : ptr_(nullptr), count_(count) {
        if (count_ == 0) return; // nothing to allocate
        cudaError_t err = cudaMalloc(&ptr_, count_ * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
        }
    }

    // Destructor: free device memory
    ~DeviceBuffer() {
        if (ptr_) {
            cudaError_t err = cudaFree(ptr_);
            if (err != cudaSuccess) {
                // In destructors we should not throw, so we log the error
                std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }

    // Disable copy semantics to avoid double-free
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Move semantics
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            // Free current resources
            if (ptr_) cudaFree(ptr_);
            // Move data
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // Access the raw device pointer
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }

    // Get number of elements
    size_t size() const { return count_; }

    // Copy data from host to device
    void copy_from_host(const std::vector<T>& host_vec) {
        if (host_vec.size() != count_) {
            throw std::runtime_error("Size mismatch in copy_from_host");
        }
        cudaError_t err = cudaMemcpy(ptr_, host_vec.data(), count_ * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy H2D failed: " + std::string(cudaGetErrorString(err)));
        }
    }

    // Copy data from device to host
    void copy_to_host(std::vector<T>& host_vec) const {
        if (host_vec.size() != count_) {
            host_vec.resize(count_);
        }
        cudaError_t err = cudaMemcpy(host_vec.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy D2H failed: " + std::string(cudaGetErrorString(err)));
        }
    }

private:
    T* ptr_;
    size_t count_;
};

/* Simple kernel that multiplies each element by 2 */
__global__ void multiply_by_two(float* d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] *= 2.0f;
    }
}

int main() {
    const int N = 10;
    std::vector<float> host_vec(N);
    for (int i = 0; i < N; ++i) {
        host_vec[i] = static_cast<float>(i);
    }

    try {
        // Allocate device buffer
        DeviceBuffer<float> dev_buf(N);

        // Copy data from host to device
        dev_buf.copy_from_host(host_vec);

        // Launch kernel
        const int threadsPerBlock = 256;
        const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        multiply_by_two<<<blocksPerGrid, threadsPerBlock>>>(dev_buf.data(), N);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(err)));
        }

        // Wait for GPU to finish
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaDeviceSynchronize failed: " + std::string(cudaGetErrorString(err)));
        }

        // Copy results back to host
        dev_buf.copy_to_host(host_vec);

        // Print results
        std::cout << "Result:" << std::endl;
        for (int i = 0; i < N; ++i) {
            std::cout << host_vec[i] << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```