```cpp
/*
Aim of the program:
Create a C++ class that encapsulates a 2D pitched device allocation, storing the pointer and pitch, and providing a safe access method.

Thinking:
- Use cudaMallocPitch for efficient memory layout.
- Store device pointer and pitch as members.
- Provide RAII: constructor allocates, destructor frees.
- Disable copy to avoid accidental double free, but allow move semantics.
- Provide helper methods to copy data to/from host with cudaMemcpy2D, checking dimensions.
- Provide getters for pointer, pitch, width, height.
- Provide device-side element access via __device__ operator() that computes offset based on pitch.
- Add a simple kernel to demonstrate usage: double each element.
- In main, allocate host array, initialize, copy to device, launch kernel, copy back, print results.
- Error checking: use cudaGetLastError and cudaMemcpy etc. Throw exceptions on failure.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

inline void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " (" << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error(msg);
    }
}

template <typename T>
class PitchedDeviceArray
{
public:
    PitchedDeviceArray(size_t width, size_t height)
        : width_(width), height_(height), d_ptr_(nullptr), pitch_(0)
    {
        size_t pitchInBytes = 0;
        cudaError_t err = cudaMallocPitch(&d_ptr_, &pitchInBytes, width_ * sizeof(T), height_);
        checkCudaError(err, "cudaMallocPitch failed");
        pitch_ = pitchInBytes;
    }

    ~PitchedDeviceArray()
    {
        if (d_ptr_)
        {
            cudaFree(d_ptr_);
        }
    }

    // Disable copy
    PitchedDeviceArray(const PitchedDeviceArray&) = delete;
    PitchedDeviceArray& operator=(const PitchedDeviceArray&) = delete;

    // Enable move
    PitchedDeviceArray(PitchedDeviceArray&& other) noexcept
        : width_(other.width_), height_(other.height_), d_ptr_(other.d_ptr_), pitch_(other.pitch_)
    {
        other.d_ptr_ = nullptr;
    }

    PitchedDeviceArray& operator=(PitchedDeviceArray&& other) noexcept
    {
        if (this != &other)
        {
            if (d_ptr_) cudaFree(d_ptr_);
            width_ = other.width_;
            height_ = other.height_;
            d_ptr_ = other.d_ptr_;
            pitch_ = other.pitch_;
            other.d_ptr_ = nullptr;
        }
        return *this;
    }

    // Getters
    T* devicePtr() const { return static_cast<T*>(d_ptr_); }
    size_t pitch() const { return pitch_; }
    size_t width() const { return width_; }
    size_t height() const { return height_; }

    // Copy data from host to device
    void copyFromHost(const T* host, size_t hostPitch)
    {
        cudaError_t err = cudaMemcpy2D(devicePtr(), pitch(),
                                       host, hostPitch,
                                       width_ * sizeof(T), height_,
                                       cudaMemcpyHostToDevice);
        checkCudaError(err, "cudaMemcpy2D HostToDevice failed");
    }

    // Copy data from device to host
    void copyToHost(T* host, size_t hostPitch) const
    {
        cudaError_t err = cudaMemcpy2D(host, hostPitch,
                                       devicePtr(), pitch(),
                                       width_ * sizeof(T), height_,
                                       cudaMemcpyDeviceToHost);
        checkCudaError(err, "cudaMemcpy2D DeviceToHost failed");
    }

    // Device side element access
    __device__ T& operator()(size_t x, size_t y)
    {
        char* row = reinterpret_cast<char*>(devicePtr()) + y * pitch_;
        return *reinterpret_cast<T*>(row + x * sizeof(T));
    }

    __device__ const T& operator()(size_t x, size_t y) const
    {
        const char* row = reinterpret_cast<const char*>(devicePtr()) + y * pitch_;
        return *reinterpret_cast<const T*>(row + x * sizeof(T));
    }

private:
    size_t width_;
    size_t height_;
    void* d_ptr_;
    size_t pitch_;
};

// Simple kernel that doubles each element
__global__ void doubleKernel(PitchedDeviceArray<float> arr)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < arr.width() && y < arr.height())
    {
        arr(x, y) *= 2.0f;
    }
}

int main()
{
    const size_t width = 5;
    const size_t height = 4;
    const size_t hostPitch = width * sizeof(float); // tightly packed host array

    // Host data
    std::vector<float> h_data(height * width);
    for (size_t i = 0; i < height; ++i)
    {
        for (size_t j = 0; j < width; ++j)
        {
            h_data[i * width + j] = static_cast<float>(i * width + j + 1);
        }
    }

    // Allocate pitched device array
    PitchedDeviceArray<float> d_arr(width, height);

    // Copy host data to device
    d_arr.copyFromHost(h_data.data(), hostPitch);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    doubleKernel<<<gridSize, blockSize>>>(d_arr);
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Kernel launch failed");
    err = cudaDeviceSynchronize();
    checkCudaError(err, "Kernel execution failed");

    // Copy back to host
    std::vector<float> h_result(height * width);
    d_arr.copyToHost(h_result.data(), hostPitch);

    // Print results
    std::cout << "Result after doubling:\n";
    for (size_t i = 0; i < height; ++i)
    {
        for (size_t j = 0; j < width; ++j)
        {
            std::cout << h_result[i * width + j] << "\t";
        }
        std::cout << "\n";
    }

    return 0;
}
```