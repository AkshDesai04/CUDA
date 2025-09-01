/*
Aim of the program: Add a member function `get()` that returns the raw device pointer.

Thinking:
- The program demonstrates a simple CUDA C++ class that manages a device array.
- The class holds a raw device pointer and its size.
- The constructor allocates device memory, and the destructor frees it.
- A member function `get()` is added to return the raw device pointer, allowing direct kernel access.
- The main function shows usage: allocate, initialize on device, copy back, and print results.
- Basic error checking is performed after CUDA calls.
- The code is fully self-contained in a single .cu file, compilable with nvcc.
*/

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << err << " \"" << cudaGetErrorString(err) \
                      << "\"\n";                                              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Simple kernel to initialize an array
__global__ void initKernel(int *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 10; // arbitrary initialization
    }
}

class DeviceArray {
public:
    // Constructor: allocate device memory for given size
    DeviceArray(int size)
        : size_(size), dev_ptr_(nullptr)
    {
        CUDA_CHECK(cudaMalloc(&dev_ptr_, size_ * sizeof(int)));
    }

    // Destructor: free device memory
    ~DeviceArray()
    {
        if (dev_ptr_) {
            cudaFree(dev_ptr_);
        }
    }

    // Delete copy constructor and assignment to avoid accidental copies
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    // Move constructor and assignment
    DeviceArray(DeviceArray&& other) noexcept
        : size_(other.size_), dev_ptr_(other.dev_ptr_)
    {
        other.dev_ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceArray& operator=(DeviceArray&& other) noexcept
    {
        if (this != &other) {
            if (dev_ptr_) cudaFree(dev_ptr_);
            dev_ptr_ = other.dev_ptr_;
            size_ = other.size_;
            other.dev_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Member function that returns the raw device pointer
    int* get() const { return dev_ptr_; }

    int size() const { return size_; }

private:
    int size_;
    int* dev_ptr_;
};

int main()
{
    const int N = 10;
    DeviceArray devArr(N);

    // Launch kernel to initialize device array
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    initKernel<<<blocks, threadsPerBlock>>>(devArr.get(), N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back to host for verification
    int hostData[N];
    CUDA_CHECK(cudaMemcpy(hostData, devArr.get(), N * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "Device array contents:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << hostData[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
