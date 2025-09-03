/*
Aim of the program: Write a comment block with best practices for avoiding memory leaks in CUDA (e.g., using RAII wrappers).

Thinking:
The developer instructions require that I produce a complete CUDA .cu file, beginning with a multiline comment that contains the exact aim statement from the user prompt, followed by my own detailed thoughts. This comment block must be followed by actual CUDA C code that can be compiled and run. Since the user only asked for a comment block about memory‑leak best practices, I will embed a small example that demonstrates RAII‑style memory management on the device side. I’ll create a tiny RAII wrapper class `DeviceVector` that automatically frees GPU memory when it goes out of scope. The main program will launch a simple kernel to show that the wrapper works. The code will compile with `nvcc` and includes all necessary headers. No external files or supporting information will be output, adhering strictly to the developer instruction to only output the .cu file content. 
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// Simple kernel that sets each element to its index
__global__ void init_kernel(int *d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_data[idx] = idx;
}

// RAII wrapper for device memory allocation
class DeviceVector {
public:
    DeviceVector(size_t size) : size_(size), ptr_(nullptr) {
        if (cudaMalloc(&ptr_, size_ * sizeof(int))) {
            throw std::runtime_error("cudaMalloc failed");
        }
    }
    ~DeviceVector() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }
    // Disable copying
    DeviceVector(const DeviceVector&) = delete;
    DeviceVector& operator=(const DeviceVector&) = delete;
    // Allow moving
    DeviceVector(DeviceVector&& other) noexcept : size_(other.size_), ptr_(other.ptr_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    DeviceVector& operator=(DeviceVector&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    int* data() const { return ptr_; }
    size_t size() const { return size_; }

private:
    size_t size_;
    int* ptr_;
};

int main() {
    const size_t N = 256;
    try {
        DeviceVector d_vec(N);

        // Launch kernel to initialize data
        int threadsPerBlock = 128;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel<<<blocks, threadsPerBlock>>>(d_vec.data(), N);
        cudaDeviceSynchronize();

        // Copy back to host and verify
        int* h_vec = new int[N];
        cudaMemcpy(h_vec, d_vec.data(), N * sizeof(int), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < N; ++i) {
            if (h_vec[i] != static_cast<int>(i)) {
                std::cerr << "Mismatch at index " << i << ": " << h_vec[i] << std::endl;
                delete[] h_vec;
                return EXIT_FAILURE;
            }
        }
        delete[] h_vec;
        std::cout << "DeviceVector RAII test passed." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
