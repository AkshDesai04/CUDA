```cpp
/*
Write a simple C++ vector-like class for device memory. The constructor calls `cudaMalloc`, the destructor calls `cudaFree`.

--- Reasoning ---
The goal is to provide a lightweight, device‑memory‑backed container that mimics the basic interface of std::vector. The container will:

1. Allocate memory on the device in its constructor via `cudaMalloc`.
2. Free that memory in its destructor via `cudaFree`.
3. Store the pointer to device memory, current size, and capacity.
4. Provide move semantics to transfer ownership without copying.
5. Disable copying to avoid accidental deep copies which would require manual implementation.
6. Offer simple operations: `data()`, `size()`, `capacity()`, `reserve()`, and `push_back()`.  
   `reserve()` reallocates device memory and copies existing elements with `cudaMemcpy`.  
   `push_back()` expands capacity when needed and copies a single element from host to device.
7. Include a small test kernel to initialize the vector and a main function to verify the implementation.

The implementation focuses on clarity and correctness over performance. For real-world use, additional features such as element access on the host side, iterator support, and more robust error handling would be desirable.

*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

/* Simple CUDA error checking macro */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__      \
                      << " (" << cudaGetErrorString(err) << ")\n";           \
            throw std::runtime_error(cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

/* DeviceVector: a vector-like container that lives in device memory */
template <typename T>
class DeviceVector {
public:
    /* Default constructor */
    DeviceVector() : d_data_(nullptr), size_(0), capacity_(0) {}

    /* Constructor that allocates memory for n elements */
    explicit DeviceVector(size_t n) : d_data_(nullptr), size_(n), capacity_(n) {
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&d_data_, n * sizeof(T)));
        }
    }

    /* Destructor frees device memory */
    ~DeviceVector() {
        if (d_data_) {
            CUDA_CHECK(cudaFree(d_data_));
        }
    }

    /* Disable copy semantics */
    DeviceVector(const DeviceVector&) = delete;
    DeviceVector& operator=(const DeviceVector&) = delete;

    /* Move constructor */
    DeviceVector(DeviceVector&& other) noexcept
        : d_data_(other.d_data_), size_(other.size_), capacity_(other.capacity_) {
        other.d_data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    /* Move assignment */
    DeviceVector& operator=(DeviceVector&& other) noexcept {
        if (this != &other) {
            // Free existing memory
            if (d_data_) {
                CUDA_CHECK(cudaFree(d_data_));
            }
            // Transfer ownership
            d_data_ = other.d_data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            // Nullify source
            other.d_data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    /* Return raw device pointer */
    T* data() { return d_data_; }
    const T* data() const { return d_data_; }

    /* Current size */
    size_t size() const { return size_; }

    /* Current capacity */
    size_t capacity() const { return capacity_; }

    /* Reserve capacity (device side) */
    void reserve(size_t new_cap) {
        if (new_cap <= capacity_) return;
        T* new_data;
        CUDA_CHECK(cudaMalloc(&new_data, new_cap * sizeof(T)));
        if (d_data_ && size_ > 0) {
            CUDA_CHECK(cudaMemcpy(new_data, d_data_, size_ * sizeof(T),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaFree(d_data_));
        }
        d_data_ = new_data;
        capacity_ = new_cap;
    }

    /* Push back a single element (host to device) */
    void push_back(const T& val) {
        if (size_ == capacity_) {
            size_t new_cap = capacity_ == 0 ? 1 : 2 * capacity_;
            reserve(new_cap);
        }
        // Copy value from host to device
        CUDA_CHECK(cudaMemcpy(d_data_ + size_, &val, sizeof(T),
                              cudaMemcpyHostToDevice));
        ++size_;
    }

private:
    T* d_data_;
    size_t size_;
    size_t capacity_;
};

/* Kernel to initialize a device array with a constant value */
__global__ void init_kernel(int* d_arr, size_t n, int val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = val;
    }
}

/* Host function to test DeviceVector */
int main() {
    const size_t N = 10;
    const int init_val = 42;

    /* Create a DeviceVector with N elements */
    DeviceVector<int> d_vec(N);

    /* Launch kernel to initialize the vector's data */
    size_t threads_per_block = 256;
    size_t blocks = (N + threads_per_block - 1) / threads_per_block;
    init_kernel<<<blocks, threads_per_block>>>(d_vec.data(), N, init_val);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy data back to host for verification */
    int h_arr[N];
    CUDA_CHECK(cudaMemcpy(h_arr, d_vec.data(), N * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "DeviceVector contents after initialization:\n";
    for (size_t i = 0; i < N; ++i) {
        std::cout << h_arr[i] << " ";
    }
    std::cout << "\n";

    /* Push back a new element */
    int new_val = 99;
    d_vec.push_back(new_val);
    std::cout << "Size after push_back: " << d_vec.size() << "\n";

    /* Copy all elements back to host */
    size_t total_size = d_vec.size();
    int* h_total = new int[total_size];
    CUDA_CHECK(cudaMemcpy(h_total, d_vec.data(), total_size * sizeof(int),
                          cudaMemcpyDeviceToHost));

    std::cout << "DeviceVector contents after push_back:\n";
    for (size_t i = 0; i < total_size; ++i) {
        std::cout << h_total[i] << " ";
    }
    std::cout << "\n";

    delete[] h_total;
    return 0;
}
```