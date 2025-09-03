```cpp
/*
Aim of the program:
Create a similar RAII wrapper for pinned host memory (`cudaMallocHost`/`cudaFreeHost`).

Thinking:
The task is to provide a resource‐managing class that encapsulates pinned host memory
allocation and deallocation, mirroring the RAII pattern used for device memory.
Pinned memory is allocated with `cudaMallocHost` and freed with `cudaFreeHost`.
The wrapper should:

1. Allocate memory in the constructor, throwing if allocation fails.
2. Release memory in the destructor.
3. Disallow copying to avoid double free.
4. Support moving to transfer ownership safely.
5. Provide convenient accessors: `data()`, `size()`, `operator[]`.
6. Be templated so it can store any POD type, typically `float`, `int`, etc.
7. Include a small demonstration (`main`) that allocates pinned memory,
   fills it, copies it to a device array, and prints a few elements to verify
   that the memory is indeed usable.

The code will also define a simple CUDA error checking macro for clarity.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

// Simple CUDA error checking macro
#define CUDA_CHECK(err)                                 \
    do {                                                \
        cudaError_t err_ = (err);                       \
        if (err_ != cudaSuccess) {                     \
            std::cerr << "CUDA error: "                \
                      << cudaGetErrorString(err_)      \
                      << " at line " << __LINE__       \
                      << " in file " << __FILE__ << std::endl; \
            throw std::runtime_error("CUDA error");     \
        }                                               \
    } while (0)

// RAII wrapper for pinned host memory
template <typename T>
class PinnedMemory {
public:
    // Constructor: allocate pinned memory for 'count' elements
    explicit PinnedMemory(size_t count)
        : ptr_(nullptr), count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMallocHost((void**)&ptr_, count_ * sizeof(T)));
        }
    }

    // Destructor: free pinned memory
    ~PinnedMemory() {
        if (ptr_) {
            CUDA_CHECK(cudaFreeHost(ptr_));
        }
    }

    // Delete copy constructor and copy assignment
    PinnedMemory(const PinnedMemory&) = delete;
    PinnedMemory& operator=(const PinnedMemory&) = delete;

    // Move constructor
    PinnedMemory(PinnedMemory&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    // Move assignment
    PinnedMemory& operator=(PinnedMemory&& other) noexcept {
        if (this != &other) {
            // Free existing memory
            if (ptr_) {
                CUDA_CHECK(cudaFreeHost(ptr_));
            }
            // Transfer ownership
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // Accessors
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return count_; }

    // Element access
    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }

private:
    T* ptr_;
    size_t count_;
};

// Kernel that simply adds 1.0f to each element
__global__
void addOneKernel(float* d_arr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1.0f;
    }
}

int main() {
    const size_t N = 10;
    try {
        // Allocate pinned host memory
        PinnedMemory<float> hostMem(N);
        // Initialize host data
        for (size_t i = 0; i < N; ++i) {
            hostMem[i] = static_cast<float>(i);
        }

        // Allocate device memory
        float* d_arr = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_arr, N * sizeof(float)));

        // Copy data from host (pinned) to device
        CUDA_CHECK(cudaMemcpy(d_arr, hostMem.data(), N * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Launch kernel
        const int threadsPerBlock = 256;
        const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        addOneKernel<<<blocks, threadsPerBlock>>>(d_arr, N);
        CUDA_CHECK(cudaGetLastError());

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(hostMem.data(), d_arr, N * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Print results
        std::cout << "Result after kernel:" << std::endl;
        for (size_t i = 0; i < N; ++i) {
            std::cout << "hostMem[" << i << "] = " << hostMem[i] << std::endl;
        }

        // Clean up device memory
        CUDA_CHECK(cudaFree(d_arr));

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```