```cpp
/*
Aim of the program:
Show how RAII (e.g., your device memory wrapper class) works well with exceptions
to ensure cudaFree is called even if an error occurs mid-function.

Thinking:
The core idea is to encapsulate device memory allocation and deallocation
in a small C++ class that follows RAII semantics: memory is allocated in the
constructor and freed in the destructor. When an exception is thrown, the
stack unwinding process will call the destructor of any local objects that
have been constructed, ensuring that cudaFree is executed.

To demonstrate this, we create a DeviceBuffer class that allocates a
cuda array on the device in its constructor and frees it in its destructor.
In the main function we allocate a buffer, perform a dummy operation that
throws an exception, and then catch the exception. By printing a message
from the destructor we can confirm that cudaFree is indeed called.

We also provide a helper macro to check CUDA errors in a concise way,
although RAII itself guarantees that resources are cleaned up regardless
of the error path. The program is written in CUDA C++ and should compile
with nvcc.*/

#include <cstdio>
#include <stdexcept>
#include <cuda_runtime.h>

// Helper macro for checking CUDA errors
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error: ") +          \
                                     cudaGetErrorString(err));             \
        }                                                                   \
    } while (0)

// RAII wrapper for device memory
class DeviceBuffer {
public:
    DeviceBuffer(size_t size) : size_(size), d_ptr_(nullptr) {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_ptr_, size_));
        printf("DeviceBuffer: Allocated %zu bytes at %p\n", size_, d_ptr_);
    }

    // Disable copy semantics
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Allow move semantics
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : size_(other.size_), d_ptr_(other.d_ptr_) {
        other.d_ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free(); // free current buffer if any
            size_ = other.size_;
            d_ptr_ = other.d_ptr_;
            other.d_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Destructor frees device memory
    ~DeviceBuffer() { free(); }

    // Accessor for device pointer
    void* get() const { return d_ptr_; }

    size_t size() const { return size_; }

private:
    void free() {
        if (d_ptr_) {
            // cudaFree is exception safe; errors are ignored here to avoid throwing from destructor
            cudaError_t err = cudaFree(d_ptr_);
            if (err != cudaSuccess) {
                fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
            } else {
                printf("DeviceBuffer: Freed %zu bytes at %p\n", size_, d_ptr_);
            }
            d_ptr_ = nullptr;
            size_ = 0;
        }
    }

    size_t size_;
    void* d_ptr_;
};

// Dummy kernel that simply copies data
__global__ void dummyKernel(int* d_in, int* d_out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx] + 1;
    }
}

int main() {
    const size_t N = 1024;
    const size_t bytes = N * sizeof(int);

    try {
        // Allocate host memory
        int* h_in = new int[N];
        int* h_out = new int[N];

        for (size_t i = 0; i < N; ++i) {
            h_in[i] = static_cast<int>(i);
        }

        // Allocate device memory using RAII
        DeviceBuffer d_in_buf(bytes);
        DeviceBuffer d_out_buf(bytes);

        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_in_buf.get(), h_in, bytes, cudaMemcpyHostToDevice));

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        dummyKernel<<<blocksPerGrid, threadsPerBlock>>>(
            static_cast<int*>(d_in_buf.get()),
            static_cast<int*>(d_out_buf.get()),
            N
        );
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());

        // Intentionally throw an exception to simulate an error in the middle of a function
        throw std::runtime_error("Simulated mid-function exception");

        // This part will not execute due to exception
        CUDA_CHECK(cudaMemcpy(h_out, d_out_buf.get(), bytes, cudaMemcpyDeviceToHost));

        // Print first few results
        for (int i = 0; i < 10; ++i) {
            printf("%d + 1 = %d\n", h_in[i], h_out[i]);
        }

        delete[] h_in;
        delete[] h_out;
    } catch (const std::exception& ex) {
        printf("Caught exception: %s\n", ex.what());
        // Even though an exception was thrown, DeviceBuffer destructors should have been called
        // freeing the device memory.
    }

    // Ensure that CUDA device is reset before exiting
    cudaDeviceReset();
    return 0;
}
```