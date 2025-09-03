```cpp
/*
Add a move constructor and a move assignment operator to allow for efficient transfers of ownership of the device pointer.

Thought process:
1. The goal is to manage a raw device pointer in a RAII class, enabling safe allocation and deallocation
   while also supporting move semantics for efficient ownership transfer.
2. Copy operations are intentionally disabled to prevent accidental deep copies of the device memory,
   which would be expensive and error-prone.
3. The move constructor steals the pointer and size from the source, sets the source's pointer to nullptr,
   and leaves the source in a valid but empty state.
4. The move assignment operator first frees any existing allocation in the destination, then steals the
   source's pointer and size, and finally nullifies the source's pointer.
5. A small kernel is provided to demonstrate usage: it doubles each element in the array.
6. In main() we create a DeviceVector, populate it with data, then move it to another instance using
   both the move constructor and the move assignment operator, verifying that the source is empty after
   each transfer.
7. All operations check CUDA error status and report failures via assert or std::cerr.
8. The code is self-contained in a single .cu file and can be compiled with nvcc.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <utility> // for std::move

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl;                  \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// A simple RAII wrapper for a device array
template <typename T>
class DeviceVector {
public:
    // Default constructor
    DeviceVector() : d_ptr_(nullptr), size_(0) {}

    // Constructor allocating device memory
    explicit DeviceVector(size_t size) : d_ptr_(nullptr), size_(size) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMalloc(&d_ptr_, size_ * sizeof(T)));
        }
    }

    // Destructor frees device memory
    ~DeviceVector() {
        if (d_ptr_) {
            CUDA_CHECK(cudaFree(d_ptr_));
        }
    }

    // Delete copy constructor and copy assignment
    DeviceVector(const DeviceVector&) = delete;
    DeviceVector& operator=(const DeviceVector&) = delete;

    // Move constructor
    DeviceVector(DeviceVector&& other) noexcept
        : d_ptr_(other.d_ptr_), size_(other.size_) {
        other.d_ptr_ = nullptr;
        other.size_ = 0;
    }

    // Move assignment operator
    DeviceVector& operator=(DeviceVector&& other) noexcept {
        if (this != &other) {
            // Free current allocation
            if (d_ptr_) {
                CUDA_CHECK(cudaFree(d_ptr_));
            }
            // Steal ownership
            d_ptr_ = other.d_ptr_;
            size_ = other.size_;
            // Leave source empty
            other.d_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Return raw device pointer
    T* data() const { return d_ptr_; }

    // Return size
    size_t size() const { return size_; }

    // Copy data from host to device
    void copy_from_host(const T* host_ptr) {
        assert(d_ptr_ && "Device memory not allocated");
        CUDA_CHECK(cudaMemcpy(d_ptr_, host_ptr, size_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Copy data from device to host
    void copy_to_host(T* host_ptr) const {
        assert(d_ptr_ && "Device memory not allocated");
        CUDA_CHECK(cudaMemcpy(host_ptr, d_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    T* d_ptr_;
    size_t size_;
};

// Simple kernel that doubles each element
__global__ void double_elements(float* d_arr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] *= 2.0f;
    }
}

int main() {
    const size_t N = 10;
    float h_data[N];
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // Create a DeviceVector and copy host data to device
    DeviceVector<float> vec1(N);
    vec1.copy_from_host(h_data);

    // Use move constructor to transfer ownership to vec2
    DeviceVector<float> vec2 = std::move(vec1);
    std::cout << "After move construction, vec1 size: " << vec1.size() << std::endl;
    std::cout << "vec2 size: " << vec2.size() << std::endl;

    // Launch kernel on vec2
    const size_t threadsPerBlock = 256;
    const size_t blocks = (vec2.size() + threadsPerBlock - 1) / threadsPerBlock;
    double_elements<<<blocks, threadsPerBlock>>>(vec2.data(), vec2.size());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Use move assignment to transfer ownership to vec3
    DeviceVector<float> vec3(20); // initially different size
    vec3 = std::move(vec2);
    std::cout << "After move assignment, vec2 size: " << vec2.size() << std::endl;
    std::cout << "vec3 size: " << vec3.size() << std::endl;

    // Retrieve data back to host
    float h_result[N];
    vec3.copy_to_host(h_result);
    std::cout << "Result after kernel (should be doubled):" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```