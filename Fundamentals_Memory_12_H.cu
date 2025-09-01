```cpp
/*
Overload the `operator[]` (note: this is advanced as it can't directly access memory, it would need to perform a single-element copy, making it very slow but a good C++ exercise).

Thoughts:
- The goal is to demonstrate overloading operator[] for a custom array class that stores data on the GPU.
- Since device memory cannot be directly accessed from host code, operator[] must copy a single element from device to host using cudaMemcpy. This is intentionally slow and is a good exercise to illustrate why such operations are discouraged in high-performance GPU code.
- We'll create a template class DeviceArray that allocates device memory, provides a const operator[] that performs a cudaMemcpy from device to host, and a non‑const operator[] that returns a proxy object. The proxy supports assignment, again performing a single-element cudaMemcpy from host to device.
- The proxy pattern keeps the syntax of arr[i] = value; and int x = arr[i]; while hiding the complexity of the copy.
- The code will also include simple error checking, a test kernel that multiplies each element by 2, and a main function that demonstrates usage.
- The program is written in C++ but compiled with nvcc (CUDA compiler) and saved as a .cu file.
*/

#include <iostream>
#include <cuda_runtime.h>

// Simple CUDA error checker
inline void checkCuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        std::cerr << msg << " : " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Kernel to double each element
template <typename T>
__global__ void doubleKernel(T* data, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2;
    }
}

template <typename T>
class DeviceArray
{
public:
    DeviceArray(size_t n) : size_(n)
    {
        checkCuda(cudaMalloc(&d_data_, size_ * sizeof(T)), "cudaMalloc failed");
    }

    ~DeviceArray()
    {
        if (d_data_) {
            cudaFree(d_data_);
        }
    }

    // Prohibit copying
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    // Allow moving
    DeviceArray(DeviceArray&& other) noexcept
        : d_data_(other.d_data_), size_(other.size_)
    {
        other.d_data_ = nullptr;
        other.size_ = 0;
    }

    DeviceArray& operator=(DeviceArray&& other) noexcept
    {
        if (this != &other) {
            if (d_data_) cudaFree(d_data_);
            d_data_ = other.d_data_;
            size_ = other.size_;
            other.d_data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Set a value at index (host -> device)
    void set(size_t idx, const T& val)
    {
        if (idx >= size_) {
            std::cerr << "Index out of bounds in set" << std::endl;
            return;
        }
        checkCuda(cudaMemcpy(d_data_ + idx, &val, sizeof(T), cudaMemcpyHostToDevice),
                  "cudaMemcpy H2D failed in set");
    }

    // Get a value at index (device -> host)
    T get(size_t idx) const
    {
        if (idx >= size_) {
            std::cerr << "Index out of bounds in get" << std::endl;
            return T();
        }
        T val;
        checkCuda(cudaMemcpy(&val, d_data_ + idx, sizeof(T), cudaMemcpyDeviceToHost),
                  "cudaMemcpy D2H failed in get");
        return val;
    }

    // Const operator[] returns value by copying from device
    T operator[](size_t idx) const
    {
        return get(idx);
    }

    // Proxy class for non-const operator[]
    class Proxy {
    public:
        Proxy(DeviceArray* arr, size_t idx) : arr_(arr), idx_(idx) {}
        // Conversion to T
        operator T() const { return arr_->get(idx_); }
        // Assignment
        Proxy& operator=(const T& val)
        {
            arr_->set(idx_, val);
            return *this;
        }
    private:
        DeviceArray* arr_;
        size_t idx_;
    };

    // Non-const operator[] returns Proxy
    Proxy operator[](size_t idx)
    {
        return Proxy(this, idx);
    }

    // Launch kernel to double all elements
    void doubleAll()
    {
        const size_t threadsPerBlock = 256;
        const size_t blocks = (size_ + threadsPerBlock - 1) / threadsPerBlock;
        doubleKernel<<<blocks, threadsPerBlock>>>(d_data_, size_);
        checkCuda(cudaDeviceSynchronize(), "Kernel launch failed");
    }

    size_t size() const { return size_; }

private:
    T* d_data_ = nullptr;
    size_t size_ = 0;
};

int main()
{
    const size_t N = 10;
    DeviceArray<int> dArr(N);

    // Initialize host array
    for (size_t i = 0; i < N; ++i) {
        dArr[i] = static_cast<int>(i); // triggers set via Proxy
    }

    // Print initial values
    std::cout << "Initial array:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << dArr[i] << " "; // triggers get via operator[]
    }
    std::cout << std::endl;

    // Double all elements on GPU
    dArr.doubleAll();

    // Print doubled values
    std::cout << "After doubling:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << dArr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```