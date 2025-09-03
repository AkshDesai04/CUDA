```cpp
/*
Aim of the program:
Overload the `operator[]` (note: this is advanced as it can't directly access memory, it would need to perform a single-element copy, making it very slow but a good C++ exercise).

Thinking:
The goal is to create a simple C++ class that represents an array stored in CUDA device memory.  
Because device memory is not directly addressable from host code, we cannot provide a normal
`operator[]` that returns a reference to an element.  Instead, we overload `operator[]` to
perform a synchronous copy of a single element from device to host using `cudaMemcpy`.  
The return type is the element type `T`, so the operator copies the value into a temporary
host variable and returns it.  A complementary `set` method is provided to write a value
back to the device using a host-to-device copy.

To demonstrate the overhead of this approach, the program:
1. Allocates a device array of integers.
2. Initializes the array on the device with a simple kernel.
3. Reads each element back to the host using the overloaded `operator[]`, printing them.
4. Modifies one element using `set`, and prints again.

The class manages its own memory, freeing it in the destructor.  All CUDA calls are wrapped
with a simple error‑checking macro.

This program is written as a single `.cu` file that can be compiled with `nvcc`.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple error checking macro
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Device array container
template <typename T>
class DeviceArray {
public:
    DeviceArray(size_t n) : size_(n), d_data_(nullptr) {
        CHECK_CUDA(cudaMalloc(&d_data_, size_ * sizeof(T)));
    }
    ~DeviceArray() {
        if (d_data_) {
            CHECK_CUDA(cudaFree(d_data_));
        }
    }

    // Overloaded operator[] that copies a single element from device to host
    T operator[](size_t idx) const {
        if (idx >= size_) {
            fprintf(stderr, "Index out of bounds in DeviceArray::operator[]\n");
            exit(EXIT_FAILURE);
        }
        T h_val;
        CHECK_CUDA(cudaMemcpy(&h_val, d_data_ + idx, sizeof(T), cudaMemcpyDeviceToHost));
        return h_val;
    }

    // Set a value at a given index (host-to-device copy)
    void set(size_t idx, const T& val) {
        if (idx >= size_) {
            fprintf(stderr, "Index out of bounds in DeviceArray::set\n");
            exit(EXIT_FAILURE);
        }
        CHECK_CUDA(cudaMemcpy(d_data_ + idx, &val, sizeof(T), cudaMemcpyHostToDevice));
    }

    // Accessor for the raw device pointer (used by kernels)
    T* data() { return d_data_; }
    const T* data() const { return d_data_; }

    size_t size() const { return size_; }

private:
    size_t size_;
    T* d_data_;
};

// Kernel to initialize the device array
__global__ void initKernel(int* d_arr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = static_cast<int>(idx * 2);  // Example initialization
    }
}

int main() {
    const size_t N = 10;
    DeviceArray<int> dev_arr(N);

    // Initialize device array using kernel
    const size_t threadsPerBlock = 256;
    const size_t blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    initKernel<<<blocks, threadsPerBlock>>>(dev_arr.data(), N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read and print elements using overloaded operator[]
    printf("Initial device array values (accessed via operator[]):\n");
    for (size_t i = 0; i < dev_arr.size(); ++i) {
        int val = dev_arr[i];  // Copies single element from device
        printf("dev_arr[%zu] = %d\n", i, val);
    }

    // Modify an element on the device
    printf("\nSetting dev_arr[3] = 42 on device.\n");
    dev_arr.set(3, 42);

    // Read and print again
    printf("Device array values after modification:\n");
    for (size_t i = 0; i < dev_arr.size(); ++i) {
        int val = dev_arr[i];
        printf("dev_arr[%zu] = %d\n", i, val);
    }

    return 0;
}
```