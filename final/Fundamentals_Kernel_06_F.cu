/*
Aim of the program:
Wrap the memory allocation and deallocation in a C++ class using RAII (Constructor allocates, Destructor frees).

Thinking:
- The user wants a CUDA program that demonstrates RAII for device memory.
- Use a C++ class (e.g., DeviceMemory<T>) that allocates device memory in its constructor
  using cudaMalloc, and frees it in its destructor with cudaFree.
- Provide a pointer accessor to allow kernels to use the memory.
- Disable copying to prevent double‑free; enable move semantics so the resource can be transferred.
- In main(), create an instance of the wrapper, allocate a simple array,
  initialize host data, copy it to device, launch a kernel that operates on it,
  copy back the result, and print it.
- No external files or additional output, just the complete .cu source code.

Below is the complete CUDA C++ program that implements this idea.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <iostream>

// Simple kernel that increments each element by 1
__global__ void incrementKernel(int* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1;
    }
}

// RAII wrapper for device memory
template<typename T>
class DeviceMemory {
public:
    // Constructor: allocate device memory
    explicit DeviceMemory(size_t count)
        : m_count(count), m_ptr(nullptr)
    {
        if (m_count == 0) {
            return;
        }
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&m_ptr), m_count * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
    }

    // Destructor: free device memory
    ~DeviceMemory()
    {
        if (m_ptr) {
            cudaFree(m_ptr);
            m_ptr = nullptr;
        }
    }

    // Disable copy semantics
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    // Enable move semantics
    DeviceMemory(DeviceMemory&& other) noexcept
        : m_count(other.m_count), m_ptr(other.m_ptr)
    {
        other.m_ptr = nullptr;
        other.m_count = 0;
    }

    DeviceMemory& operator=(DeviceMemory&& other) noexcept
    {
        if (this != &other) {
            // Free existing resource
            if (m_ptr) {
                cudaFree(m_ptr);
            }
            m_ptr = other.m_ptr;
            m_count = other.m_count;
            other.m_ptr = nullptr;
            other.m_count = 0;
        }
        return *this;
    }

    // Accessor for raw pointer
    T* get() const { return m_ptr; }

    // Size accessor
    size_t size() const { return m_count; }

private:
    size_t m_count;
    T*    m_ptr;
};

int main()
{
    const size_t N = 10;
    std::vector<int> hostVec(N);
    for (size_t i = 0; i < N; ++i) {
        hostVec[i] = static_cast<int>(i);
    }

    try {
        // Allocate device memory using RAII
        DeviceMemory<int> devMem(N);

        // Copy data from host to device
        cudaMemcpy(devMem.get(), hostVec.data(), N * sizeof(int), cudaMemcpyHostToDevice);

        // Launch kernel
        const int threadsPerBlock = 256;
        const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        incrementKernel<<<blocks, threadsPerBlock>>>(devMem.get(), N);
        cudaDeviceSynchronize();

        // Copy result back to host
        cudaMemcpy(hostVec.data(), devMem.get(), N * sizeof(int), cudaMemcpyDeviceToHost);

        // Print result
        std::cout << "Result after kernel:\n";
        for (size_t i = 0; i < N; ++i) {
            std::cout << hostVec[i] << " ";
        }
        std::cout << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
