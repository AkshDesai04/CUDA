/*
Aim: Add a member function `get()` that returns the raw device pointer.

Thinking:  
- Implement a small RAII wrapper around CUDA device memory (`cudaMalloc`/`cudaFree`).  
- Provide a member function `get()` that returns the raw `T*` device pointer so callers can use it directly when launching kernels or performing other CUDA API calls.  
- Include constructors, destructor, copy/move semantics, and a simple size accessor.  
- Add a small test in `main()` that allocates a device array, fills it on the host, copies it to the device, uses `get()` to launch a simple kernel, and copies back the result for verification.  
- Use a simple error‑checking macro (`CUDA_CHECK`) to keep the example readable.  
- The code is self‑contained, compiles as a `.cu` file with `nvcc`, and demonstrates the requested `get()` functionality.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>

/* Simple macro for CUDA error checking */
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            std::cerr << "CUDA error in " << __FILE__ << ":"          \
                      << __LINE__ << " : " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error(cudaGetErrorString(err));       \
        }                                                             \
    } while (0)

/* DeviceArray: RAII wrapper around device memory */
template <typename T>
class DeviceArray {
public:
    /* Constructors */
    explicit DeviceArray(size_t n)
        : size_(n), ptr_(nullptr) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, size_ * sizeof(T)));
        }
    }

    /* Destructor */
    ~DeviceArray() {
        if (ptr_) {
            CUDA_CHECK(cudaFree(ptr_));
        }
    }

    /* Delete copy constructor/assignment */
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    /* Move constructor */
    DeviceArray(DeviceArray&& other) noexcept
        : size_(other.size_), ptr_(other.ptr_) {
        other.size_ = 0;
        other.ptr_  = nullptr;
    }

    /* Move assignment */
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            if (ptr_) CUDA_CHECK(cudaFree(ptr_));
            size_ = other.size_;
            ptr_  = other.ptr_;
            other.size_ = 0;
            other.ptr_  = nullptr;
        }
        return *this;
    }

    /* Accessors */
    size_t size() const { return size_; }

    /* get() returns the raw device pointer */
    T* get() const { return ptr_; }

    /* Data transfer helpers */
    void copy_from_host(const T* host_ptr) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMemcpy(ptr_, host_ptr, size_ * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    void copy_to_host(T* host_ptr) const {
        if (size_ > 0) {
            CUDA_CHECK(cudaMemcpy(host_ptr, ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

private:
    size_t size_;
    T*    ptr_;
};

/* Simple kernel: increment each element by one */
__global__
void increment_kernel(int* d_data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] += 1;
    }
}

int main() {
    const size_t N = 10;
    std::vector<int> host_vec(N);
    for (size_t i = 0; i < N; ++i) host_vec[i] = static_cast<int>(i);

    // Allocate device array
    DeviceArray<int> dev_arr(N);

    // Copy data from host to device
    dev_arr.copy_from_host(host_vec.data());

    // Launch kernel using get() to obtain raw pointer
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_arr.get(), N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    std::vector<int> result(N);
    dev_arr.copy_to_host(result.data());

    // Print results
    std::cout << "Result after kernel:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
