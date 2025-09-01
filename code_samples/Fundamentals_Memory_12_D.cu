/*
Add member functions `copy_to_device(const T* host_ptr, size_t count)` and `copy_to_host(T* host_ptr, size_t count)`.

The goal of this program is to demonstrate how to encapsulate CUDA device memory management within a
templated C++ class and provide convenient methods for copying data between host and device. The class
`DeviceArray<T>` owns a pointer to device memory and the number of elements it can hold. It allocates
memory on construction, frees it on destruction, and exposes two member functions:

- `copy_to_device(const T* host_ptr, size_t count)` copies up to `count` elements from the host
  array `host_ptr` into the device array, performing bounds checking so we never copy more than the
  allocated device size.
- `copy_to_host(T* host_ptr, size_t count)` copies up to `count` elements from the device array
  back into the host array `host_ptr`, again respecting the allocated device size.

To keep the example self‑contained we also provide a simple kernel that squares each element of the
array. A test harness allocates a host array, initializes it, copies it to the device, runs the
kernel, copies the result back, and prints it. Error handling is done via a macro that prints the
CUDA error string and aborts if a CUDA API call fails. The program is written in C++ but compiled
with `nvcc` as a CUDA source file (`.cu`). All the code is inside a single source file.

The key design decisions are:
- Use a template class so the same code works for any arithmetic type (float, double, int, etc.).
- Perform size checks in the copy functions to avoid overruns and to provide clear feedback.
- Keep the interface minimal but useful for typical host‑to‑device and device‑to‑host workflows.
- Demonstrate usage with a simple kernel that operates on the same templated type.

Below is the full code. Compile with `nvcc -arch=sm_70 -o device_array device_array.cu` (adjust `-arch` as needed).
*/

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

// Macro to check CUDA API return codes
#define CUDA_CHECK(call)                                           \
    {                                                              \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":"       \
                      << __LINE__ << " : "                         \
                      << cudaGetErrorString(err) << std::endl;     \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    }

// Templated device array class
template <typename T>
class DeviceArray {
public:
    DeviceArray(size_t n) : size(n), d_ptr(nullptr) {
        CUDA_CHECK(cudaMalloc(&d_ptr, size * sizeof(T)));
    }

    ~DeviceArray() {
        if (d_ptr) {
            cudaFree(d_ptr);
        }
    }

    // Copy from host to device
    void copy_to_device(const T* host_ptr, size_t count) {
        size_t to_copy = (count <= size) ? count : size;
        CUDA_CHECK(cudaMemcpy(d_ptr, host_ptr,
                              to_copy * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    // Copy from device to host
    void copy_to_host(T* host_ptr, size_t count) {
        size_t to_copy = (count <= size) ? count : size;
        CUDA_CHECK(cudaMemcpy(host_ptr, d_ptr,
                              to_copy * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }

    T* device_ptr() const { return d_ptr; }
    size_t size_bytes() const { return size * sizeof(T); }

private:
    T* d_ptr;
    size_t size;
};

// Templated kernel that squares each element
template <typename T>
__global__ void squareKernel(T* d_data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] = d_data[idx] * d_data[idx];
    }
}

// Helper to launch the kernel
template <typename T>
void launch_square_kernel(DeviceArray<T>& d_array) {
    const size_t threads_per_block = 256;
    size_t blocks = (d_array.size_bytes() / sizeof(T) + threads_per_block - 1) / threads_per_block;
    squareKernel<T><<<blocks, threads_per_block>>>(d_array.device_ptr(),
                                                   d_array.size_bytes() / sizeof(T));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main() {
    const size_t N = 10;
    float h_data[N];

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i + 1); // 1.0, 2.0, ...
    }

    std::cout << "Host data before kernel:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Allocate device array
    DeviceArray<float> d_array(N);

    // Copy data to device
    d_array.copy_to_device(h_data, N);

    // Launch kernel to square each element
    launch_square_kernel(d_array);

    // Copy result back to host
    d_array.copy_to_host(h_data, N);

    std::cout << "Host data after kernel:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
