/*
Make the class a template `template <typename T>` so it can manage any data type.

Thinking:
- The goal is to illustrate how a templated class can be used with CUDA to handle
  arbitrary data types.
- We'll implement a simple `DeviceArray<T>` that allocates memory on the host and
  device, and provides helper methods for copying data between them.
- A templated CUDA kernel `addKernel<T>` will add a constant value to each element
  of the array.  Because the kernel itself is templated, the same code works
  for any type that supports addition with the provided constant.
- In `main` we instantiate `DeviceArray<int>` and `DeviceArray<float>` to
  demonstrate usage.  Each instance is filled, copied to the GPU, processed,
  copied back, and printed.
- CUDA error checking is performed via a macro to keep the example clean.
- The code is self‑contained and should compile with any recent CUDA toolkit.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                      << " at line " << __LINE__ << std::endl;             \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

// Templated device array class
template <typename T>
class DeviceArray {
public:
    DeviceArray(size_t n) : size_(n), hostPtr_(nullptr), devicePtr_(nullptr) {
        hostPtr_ = new T[size_];
        CUDA_CHECK(cudaMalloc(&devicePtr_, size_ * sizeof(T)));
    }

    ~DeviceArray() {
        delete[] hostPtr_;
        CUDA_CHECK(cudaFree(devicePtr_));
    }

    // Access host memory
    T* host() { return hostPtr_; }

    // Get device pointer
    T* device() const { return devicePtr_; }

    size_t size() const { return size_; }

    // Copy data from host to device
    void copyToDevice() const {
        CUDA_CHECK(cudaMemcpy(devicePtr_, hostPtr_, size_ * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    // Copy data from device to host
    void copyFromDevice() const {
        CUDA_CHECK(cudaMemcpy(hostPtr_, devicePtr_, size_ * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }

private:
    size_t size_;
    T* hostPtr_;
    T* devicePtr_;
};

// Templated CUDA kernel that adds a constant value to each element
template <typename T>
__global__ void addKernel(T* data, T val, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += val;
    }
}

int main() {
    const size_t N = 10;

    // Integer example
    DeviceArray<int> intArray(N);
    for (size_t i = 0; i < N; ++i) {
        intArray.host()[i] = static_cast<int>(i);
    }
    intArray.copyToDevice();

    int intVal = 5;
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<int><<<blocks, threadsPerBlock>>>(intArray.device(), intVal, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    intArray.copyFromDevice();

    std::cout << "Integer array after adding " << intVal << ":\n";
    for (size_t i = 0; i < N; ++i) {
        std::cout << intArray.host()[i] << " ";
    }
    std::cout << std::endl;

    // Floating point example
    DeviceArray<float> floatArray(N);
    for (size_t i = 0; i < N; ++i) {
        floatArray.host()[i] = static_cast<float>(i) * 1.1f;
    }
    floatArray.copyToDevice();

    float floatVal = 2.5f;
    addKernel<float><<<blocks, threadsPerBlock>>>(floatArray.device(), floatVal, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    floatArray.copyFromDevice();

    std::cout << "Float array after adding " << floatVal << ":\n";
    for (size_t i = 0; i < N; ++i) {
        std::cout << floatArray.host()[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
