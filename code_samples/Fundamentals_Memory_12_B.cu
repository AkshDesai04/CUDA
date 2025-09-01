```cpp
/*
Add a move constructor and a move assignment operator to allow for efficient transfers of ownership of the device pointer.

Thinking:
- We will implement a simple RAII wrapper `DeviceArray` that manages a pointer to device memory.
- The class will provide constructors, destructor, copy operations, and crucially the move constructor and move assignment operator.
- The move operations will transfer ownership of the underlying device pointer, setting the source object's pointer to `nullptr` and size to zero.
- We will also provide helper methods to fill the array from the host and to retrieve data back to the host for verification.
- A small kernel `addOne` will be used to modify device data.
- In `main`, we will demonstrate creating a `DeviceArray`, initializing it, launching the kernel, moving it to another object, and confirming that the original object no longer owns the memory.
- All CUDA API calls will be checked for errors, and we will use `cudaMemcpy` for data transfer.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>

inline void checkCuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        std::cerr << msg << " Error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

class DeviceArray
{
public:
    // Default constructor
    DeviceArray() noexcept : d_data(nullptr), size(0) {}

    // Constructor that allocates device memory
    explicit DeviceArray(size_t n) : d_data(nullptr), size(n)
    {
        if (size == 0) return;
        checkCuda(cudaMalloc(&d_data, size * sizeof(int)), "cudaMalloc failed");
    }

    // Destructor
    ~DeviceArray()
    {
        if (d_data) {
            cudaFree(d_data);
        }
    }

    // Copy constructor
    DeviceArray(const DeviceArray& other) : d_data(nullptr), size(other.size)
    {
        if (size == 0) return;
        checkCuda(cudaMalloc(&d_data, size * sizeof(int)), "cudaMalloc failed (copy)");
        checkCuda(cudaMemcpy(d_data, other.d_data, size * sizeof(int), cudaMemcpyDeviceToDevice),
                  "cudaMemcpy failed (copy)");
    }

    // Copy assignment
    DeviceArray& operator=(const DeviceArray& other)
    {
        if (this == &other) return *this;
        // Free current resources
        if (d_data) cudaFree(d_data);
        size = other.size;
        if (size == 0) {
            d_data = nullptr;
            return *this;
        }
        checkCuda(cudaMalloc(&d_data, size * sizeof(int)), "cudaMalloc failed (copy assign)");
        checkCuda(cudaMemcpy(d_data, other.d_data, size * sizeof(int), cudaMemcpyDeviceToDevice),
                  "cudaMemcpy failed (copy assign)");
        return *this;
    }

    // Move constructor
    DeviceArray(DeviceArray&& other) noexcept
        : d_data(other.d_data), size(other.size)
    {
        other.d_data = nullptr;
        other.size = 0;
    }

    // Move assignment
    DeviceArray& operator=(DeviceArray&& other) noexcept
    {
        if (this == &other) return *this;
        // Free current resources
        if (d_data) cudaFree(d_data);
        // Steal resources
        d_data = other.d_data;
        size   = other.size;
        // Nullify source
        other.d_data = nullptr;
        other.size   = 0;
        return *this;
    }

    // Fill device array from host vector
    void fill(const std::vector<int>& hostVec)
    {
        if (hostVec.size() != size) {
            throw std::runtime_error("Size mismatch in fill()");
        }
        checkCuda(cudaMemcpy(d_data, hostVec.data(), size * sizeof(int), cudaMemcpyHostToDevice),
                  "cudaMemcpy failed (fill)");
    }

    // Retrieve device array to host vector
    std::vector<int> toHost() const
    {
        std::vector<int> hostVec(size);
        if (size > 0) {
            checkCuda(cudaMemcpy(hostVec.data(), d_data, size * sizeof(int), cudaMemcpyDeviceToHost),
                      "cudaMemcpy failed (toHost)");
        }
        return hostVec;
    }

    // Accessors
    int* data() const { return d_data; }
    size_t getSize() const { return size; }

private:
    int* d_data;
    size_t size;
};

// Simple kernel that adds 1 to each element
__global__ void addOne(int* d, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d[idx] += 1;
    }
}

int main()
{
    try {
        const size_t N = 10;
        std::vector<int> init(N);
        for (size_t i = 0; i < N; ++i) init[i] = static_cast<int>(i);

        // Create DeviceArray and fill it
        DeviceArray a(N);
        a.fill(init);

        // Launch kernel
        const int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        addOne<<<blocks, threadsPerBlock>>>(a.data(), a.getSize());
        checkCuda(cudaGetLastError(), "Kernel launch failed");
        checkCuda(cudaDeviceSynchronize(), "Kernel execution failed");

        // Retrieve and print results
        auto hostResult = a.toHost();
        std::cout << "Result after kernel on 'a': ";
        for (int v : hostResult) std::cout << v << " ";
        std::cout << std::endl;

        // Move 'a' into 'b'
        DeviceArray b = std::move(a);
        std::cout << "'a' after move: size=" << a.getSize()
                  << ", data pointer=" << static_cast<void*>(a.data()) << std::endl;
        std::cout << "'b' after move: size=" << b.getSize()
                  << ", data pointer=" << static_cast<void*>(b.data()) << std::endl;

        // Verify 'b' contains the correct data
        auto hostResultB = b.toHost();
        std::cout << "Result in 'b': ";
        for (int v : hostResultB) std::cout << v << " ";
        std::cout << std::endl;

        // Move assignment: move 'b' into 'c'
        DeviceArray c(5); // some dummy size
        c = std::move(b);
        std::cout << "'b' after move assignment: size=" << b.getSize()
                  << ", data pointer=" << static_cast<void*>(b.data()) << std::endl;
        std::cout << "'c' after move assignment: size=" << c.getSize()
                  << ", data pointer=" << static_cast<void*>(c.data()) << std::endl;

        auto hostResultC = c.toHost();
        std::cout << "Result in 'c': ";
        for (int v : hostResultC) std::cout << v << " ";
        std::cout << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
```