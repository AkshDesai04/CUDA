/*
Create a similar RAII wrapper for pinned host memory (`cudaMallocHost`/`cudaFreeHost`).

My approach:
- Define a template class `PinnedMemory<T>` that stores a pointer to `T` and a size (number of elements).
- In the constructor, allocate pinned memory using `cudaMallocHost`, checking for success.
- In the destructor, free the memory using `cudaFreeHost` if the pointer is not null.
- Delete copy constructor/assignment to avoid accidental copies.
- Implement move semantics to allow transferring ownership.
- Provide convenient methods: `data()` to get the raw pointer, `size()` to get the number of elements, `operator[]`, and range-based loop helpers.
- Add a simple `CHECK_CUDA` macro for error checking.
- In `main`, demonstrate usage by allocating pinned memory, initializing it, copying to device memory, copying back, and verifying the data.
- The RAII wrapper ensures that pinned memory is properly freed when the object goes out of scope, preventing leaks and simplifying resource management.

This program is self‑contained, compiles with `nvcc`, and runs on a system with CUDA installed.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <iomanip>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            throw std::runtime_error(                              \
                std::string("CUDA error: ") + cudaGetErrorString(err)); \
        }                                                           \
    } while (0)

// RAII wrapper for pinned host memory
template <typename T>
class PinnedMemory {
public:
    // Constructor: allocate pinned memory for `n` elements
    explicit PinnedMemory(std::size_t n)
        : ptr_(nullptr), size_(n) {
        if (n == 0) {
            return;
        }
        CHECK_CUDA(cudaMallocHost(&ptr_, n * sizeof(T)));
    }

    // Destructor: free pinned memory
    ~PinnedMemory() {
        if (ptr_) {
            CHECK_CUDA(cudaFreeHost(ptr_));
        }
    }

    // Delete copy operations
    PinnedMemory(const PinnedMemory&) = delete;
    PinnedMemory& operator=(const PinnedMemory&) = delete;

    // Move constructor
    PinnedMemory(PinnedMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    // Move assignment
    PinnedMemory& operator=(PinnedMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                CHECK_CUDA(cudaFreeHost(ptr_));
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Accessors
    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return size_; }

    // Element access
    T& operator[](std::size_t idx) {
        return ptr_[idx];
    }
    const T& operator[](std::size_t idx) const {
        return ptr_[idx];
    }

    // Range helpers for range-based loops
    T* begin() noexcept { return ptr_; }
    T* end() noexcept { return ptr_ + size_; }
    const T* begin() const noexcept { return ptr_; }
    const T* end() const noexcept { return ptr_ + size_; }

private:
    T* ptr_;
    std::size_t size_;
};

int main() {
    const std::size_t N = 10;
    try {
        // Allocate pinned host memory
        PinnedMemory<float> hostMem(N);
        // Initialize host memory
        std::generate(hostMem.begin(), hostMem.end(),
                      [n = 0]() mutable { return static_cast<float>(n++); });

        // Allocate device memory
        float* devPtr = nullptr;
        CHECK_CUDA(cudaMalloc(&devPtr, N * sizeof(float)));

        // Copy from pinned host to device
        CHECK_CUDA(cudaMemcpy(devPtr, hostMem.data(),
                              N * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Optional: perform device operations here (omitted for brevity)

        // Copy back from device to a regular host array
        float* hostBack = new float[N];
        CHECK_CUDA(cudaMemcpy(hostBack, devPtr,
                              N * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Verify the data
        std::cout << "Data copied back from device:\n";
        for (std::size_t i = 0; i < N; ++i) {
            std::cout << std::fixed << std::setprecision(1) << hostBack[i] << ' ';
        }
        std::cout << '\n';

        // Clean up
        delete[] hostBack;
        CHECK_CUDA(cudaFree(devPtr));
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    // At this point, `hostMem` goes out of scope and its destructor
    // frees the pinned memory automatically.

    return EXIT_SUCCESS;
}
