```cpp
/*
 * Aim of the program (exactly as passed in the prompt):
 * 
 * Overload the operator[] (note: this is advanced as it can't directly access memory, it would need to perform a single-element copy, making it very slow but a good C++ exercise).
 *
 * --------------------------------------------------------------------
 * Thinking:
 * 
 * The requirement is to overload the subscript operator `operator[]` for a
 * container that stores its elements in CUDA device memory.  Because the
 * host code cannot dereference a raw device pointer, accessing a single
 * element must involve copying it from the device to the host with
 * `cudaMemcpy`.  This is deliberately slow, but it serves as a good
 * exercise in mixing C++ operator overloading with CUDA memory
 * management.
 *
 * Design:
 *   • Create a simple templated class `GPUArray<T>` that owns a
 *     contiguous device buffer (`T* d_ptr`) and knows its size.
 *   • Provide a constructor that allocates device memory with
 *     `cudaMalloc` and a destructor that frees it with
 *     `cudaFree`.
 *   • Implement a `fill` method to copy a host array into the device
 *     buffer (one-shot copy).
 *   • Overload `operator[]` as a *host* function returning `T`.  The
 *     function will perform a `cudaMemcpy` of a single element from
 *     device to host, then return the value.  The const version does
 *     the same.
 *   • Include a small `main` demonstrating the usage: create an
 *     array, fill it, and access elements through `operator[]`.
 *
 * Notes:
 *   – The operator overload returns by value because we cannot return
 *     a reference to a device memory location from host code.
 *   – All CUDA calls are wrapped in a simple error‑checking macro.
 *   – The code is a complete `.cu` file ready to compile with `nvcc
 *     gpu_array_example.cu -o gpu_array_example`.
 *   – The program prints each element accessed via `operator[]`
 *     confirming the round‑trip copy.
 *
 * --------------------------------------------------------------------
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

/* Simple CUDA error checking macro */
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

/* Templated GPUArray class */
template <typename T>
class GPUArray {
public:
    GPUArray(size_t n) : size_(n), d_ptr_(nullptr) {
        CUDA_CHECK(cudaMalloc(&d_ptr_, size_ * sizeof(T)));
    }

    ~GPUArray() {
        if (d_ptr_) {
            CUDA_CHECK(cudaFree(d_ptr_));
        }
    }

    /* Disable copy semantics to avoid accidental double frees */
    GPUArray(const GPUArray&) = delete;
    GPUArray& operator=(const GPUArray&) = delete;

    /* Move semantics */
    GPUArray(GPUArray&& other) noexcept
        : size_(other.size_), d_ptr_(other.d_ptr_) {
        other.d_ptr_ = nullptr;
        other.size_ = 0;
    }

    GPUArray& operator=(GPUArray&& other) noexcept {
        if (this != &other) {
            if (d_ptr_) CUDA_CHECK(cudaFree(d_ptr_));
            d_ptr_ = other.d_ptr_;
            size_ = other.size_;
            other.d_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /* Fill device buffer from host array */
    void fill(const T* host_ptr) {
        CUDA_CHECK(cudaMemcpy(d_ptr_, host_ptr,
                              size_ * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    /* Overloaded operator[] that copies a single element from device to host */
    T operator[](size_t idx) const {
        if (idx >= size_) {
            fprintf(stderr, "Index out of bounds: %zu\n", idx);
            exit(EXIT_FAILURE);
        }
        T value;
        CUDA_CHECK(cudaMemcpy(&value, d_ptr_ + idx, sizeof(T),
                              cudaMemcpyDeviceToHost));
        return value;
    }

    /* Size accessor */
    size_t size() const { return size_; }

private:
    size_t size_;
    T*    d_ptr_;
};

/* Example usage */
int main() {
    const size_t N = 10;
    int host_data[N];
    for (size_t i = 0; i < N; ++i) host_data[i] = static_cast<int>(i * 10);

    GPUArray<int> gpu_arr(N);
    gpu_arr.fill(host_data);

    std::cout << "Elements accessed via overloaded operator[]:" << std::endl;
    for (size_t i = 0; i < gpu_arr.size(); ++i) {
        int val = gpu_arr[i];            // invokes the overloaded operator[]
        std::cout << "gpu_arr[" << i << "] = " << val << std::endl;
    }

    /* Ensure all device work is finished before exiting */
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
```