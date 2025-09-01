/*
Show how RAII (e.g., your device memory wrapper class) works well with exceptions to ensure `cudaFree` is called even if an error occurs mid-function.

Thinking:
- The goal is to demonstrate that a C++ RAII wrapper for device memory will automatically release the CUDA allocation in its destructor even if an exception interrupts the function.
- We'll create a simple template class `DeviceMemory<T>` that allocates with `cudaMalloc` in the constructor and frees with `cudaFree` in the destructor.
- The constructor will throw a `std::runtime_error` if `cudaMalloc` fails, ensuring that partially constructed objects are not leaked.
- In a test function `process()`, we allocate several device buffers and deliberately throw an exception before launching any kernel.
- Because the exception propagates out of `process()`, the stack is unwound and the `DeviceMemory` objects are destroyed, invoking their destructors, which call `cudaFree`.
- The destructor prints a message when freeing so we can see that `cudaFree` was indeed called.
- The main function calls `process()` inside a `try`/`catch` block, prints the caught exception, and then exits cleanly.
- We also provide a CUDA error-checking macro that throws on failure, which further illustrates exception safety in CUDA API calls.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdio>

// CUDA error checking helper that throws exceptions
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::string msg = std::string("CUDA error: ") + cudaGetErrorString(code)
                         + " at " + file + ":" + std::to_string(line);
        if (abort) {
            throw std::runtime_error(msg);
        } else {
            std::cerr << msg << std::endl;
        }
    }
}

#define CUDA_CHECK(ans) gpuAssert((ans), __FILE__, __LINE__)

// RAII wrapper for device memory
template <typename T>
class DeviceMemory {
public:
    explicit DeviceMemory(size_t count)
        : ptr_(nullptr), count_(count) {
        cudaError_t err = cudaMalloc(&ptr_, count_ * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed: ") +
                                     cudaGetErrorString(err));
        }
        std::cout << "DeviceMemory allocated: " << ptr_
                  << " of size " << count_ * sizeof(T) << " bytes\n";
    }

    // Delete copy semantics to avoid accidental double free
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    // Move semantics
    DeviceMemory(DeviceMemory&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            free(); // free current resource
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    ~DeviceMemory() {
        free();
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }

private:
    void free() {
        if (ptr_) {
            cudaError_t err = cudaFree(ptr_);
            if (err != cudaSuccess) {
                std::cerr << "cudaFree failed: " << cudaGetErrorString(err)
                          << std::endl;
            } else {
                std::cout << "DeviceMemory freed: " << ptr_ << std::endl;
            }
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* ptr_;
    size_t count_;
};

// Simple kernel that adds two vectors
__global__ void addKernel(const int* a, const int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Function that demonstrates exception handling with RAII
void process() {
    const int N = 10;
    // Allocate device memory
    DeviceMemory<int> d_a(N);
    DeviceMemory<int> d_b(N);
    DeviceMemory<int> d_c(N);

    // Prepare host data
    std::vector<int> h_a(N, 1);
    std::vector<int> h_b(N, 2);

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Simulate an error before launching the kernel
    throw std::runtime_error("Simulated error before kernel launch");

    // The following code would normally run, but it's unreachable in this example
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocks, threadsPerBlock>>>(d_a.get(), d_b.get(), d_c.get(), N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    std::vector<int> h_c(N);
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c.get(), N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print result
    std::cout << "Result: ";
    for (int val : h_c) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main() {
    try {
        process();
    } catch (const std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }

    // Ensure device is reset before exiting
    cudaDeviceReset();
    return 0;
}
