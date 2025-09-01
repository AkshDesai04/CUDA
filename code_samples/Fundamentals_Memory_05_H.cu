/*
Aim: Write a simple C++ vector-like class for device memory. The constructor calls cudaMalloc, the destructor calls cudaFree.

Thinking:
- Need a template class that manages a device pointer and its size.
- Constructor will allocate memory on the device via cudaMalloc.
- Destructor will free the allocated memory with cudaFree.
- Provide element access through operator[] that works both on host and device (hence __host__ __device__).
- Provide size() accessor.
- Disable copying to avoid double free; implement move semantics.
- Provide helper methods to copy data between host and device.
- Add a small kernel example to demonstrate usage.
- Ensure all CUDA API calls are checked for errors.
- Compile with nvcc and include <cuda_runtime.h> and <iostream> for host side I/O.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <utility>

/* Error checking macro */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    \
                      << " : " << cudaGetErrorString(err) << std::endl;      \
            throw std::runtime_error(cudaGetErrorString(err));              \
        }                                                                    \
    } while (0)

/* DeviceVector: a simple vector-like container for device memory */
template <typename T>
class DeviceVector {
public:
    /* Default constructor */
    __host__ DeviceVector() noexcept : data_(nullptr), size_(0) {}

    /* Size constructor: allocate memory for n elements */
    __host__ explicit DeviceVector(std::size_t n) noexcept : data_(nullptr), size_(n) {
        if (n == 0) {
            data_ = nullptr;
            return;
        }
        CUDA_CHECK(cudaMalloc(&data_, n * sizeof(T)));
    }

    /* Destructor: free device memory */
    __host__ ~DeviceVector() {
        if (data_) {
            CUDA_CHECK(cudaFree(data_));
        }
    }

    /* Delete copy constructor and copy assignment to avoid double free */
    __host__ DeviceVector(const DeviceVector&) = delete;
    __host__ DeviceVector& operator=(const DeviceVector&) = delete;

    /* Move constructor */
    __host__ DeviceVector(DeviceVector&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    /* Move assignment */
    __host__ DeviceVector& operator=(DeviceVector&& other) noexcept {
        if (this != &other) {
            // Free current resources
            if (data_) {
                CUDA_CHECK(cudaFree(data_));
            }
            // Steal resources
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /* Element access (host and device) */
    __host__ __device__ T& operator[](std::size_t idx) noexcept { return data_[idx]; }
    __host__ __device__ const T& operator[](std::size_t idx) const noexcept { return data_[idx]; }

    /* Size accessor */
    __host__ std::size_t size() const noexcept { return size_; }

    /* Pointer accessor (host only) */
    __host__ T* data() noexcept { return data_; }
    __host__ const T* data() const noexcept { return data_; }

    /* Copy data from host to device */
    __host__ void copy_from_host(const T* host_ptr) {
        if (!data_) return;
        CUDA_CHECK(cudaMemcpy(data_, host_ptr, size_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    /* Copy data from device to host */
    __host__ void copy_to_host(T* host_ptr) const {
        if (!data_) return;
        CUDA_CHECK(cudaMemcpy(host_ptr, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    T* data_;
    std::size_t size_;
};

/* Simple kernel that sets all elements to a given value */
template <typename T>
__global__ void set_kernel(T* data, std::size_t n, T value) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

/* Example usage */
int main() {
    const std::size_t N = 10;
    DeviceVector<float> devVec(N);

    // Set device vector to 3.14 using kernel
    const std::size_t threadsPerBlock = 256;
    const std::size_t blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    set_kernel<float><<<blocks, threadsPerBlock>>>(devVec.data(), devVec.size(), 3.14f);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back to host and print
    float hostData[N];
    devVec.copy_to_host(hostData);

    std::cout << "DeviceVector contents:" << std::endl;
    for (std::size_t i = 0; i < N; ++i) {
        std::cout << hostData[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
