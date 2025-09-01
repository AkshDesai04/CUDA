```cpp
/*
Aim of the program:
Write a comment block with best practices for avoiding memory leaks in CUDA (e.g., using RAII wrappers).

Thinking process:
1. The user explicitly requested a comment block that explains best practices for avoiding memory leaks in CUDA, including the use of RAII wrappers. 
2. The developer instruction overrides the user instruction and requires that the entire response be a CUDA source file (.cu). The file must start with a multiline comment that contains both the aim of the program (exactly as the user provided) and the entire thought process that led to the design of the code. 
3. After the comment, the actual CUDA code should be included. The code should demonstrate:
   - Allocation of device memory with cudaMalloc.
   - A lightweight RAII wrapper that frees the device memory in its destructor.
   - A small kernel launch to illustrate usage.
   - Proper error checking after CUDA API calls.
4. We’ll provide a small, self‑contained example that compiles with `nvcc`. The example will:
   - Allocate an array on the device.
   - Fill it with a kernel.
   - Copy the result back to host.
   - Use the RAII wrapper to automatically release the device memory.
5. The comment will include best‑practice pointers such as:
   - Always pair cudaMalloc with cudaFree.
   - Use RAII wrappers or smart pointers to automate cleanup.
   - Check CUDA errors after each API call.
   - Avoid dangling pointers by ensuring ownership semantics.
   - Prefer `cudaMemset` and `cudaMemcpy` error checks.
   - Keep device memory allocation size aligned if necessary.
   - Consider `cudaDeviceSynchronize` for debugging.
6. The final output will be a single .cu file containing the comment block and the complete code.

*/

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__   \
                      << " (" << cudaGetErrorString(err) << ")\n";        \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

/* RAII wrapper for device memory */
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t size)
        : ptr_(nullptr), size_(size) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, size_));
        }
    }

    /* Disable copy semantics */
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    /* Enable move semantics */
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /* Destructor releases device memory */
    ~DeviceBuffer() {
        release();
    }

    /* Raw pointer access */
    void* get() const { return ptr_; }

    /* Size accessor */
    size_t size() const { return size_; }

private:
    void release() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }

    void* ptr_;
    size_t size_;
};

/* Simple kernel that sets each element to its index */
__global__ void init_kernel(int* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = idx;
    }
}

int main() {
    const int N = 10;
    const size_t bytes = N * sizeof(int);

    /* Allocate device memory with RAII wrapper */
    DeviceBuffer d_buf(bytes);

    /* Launch kernel to initialize data */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    init_kernel<<<blocksPerGrid, threadsPerBlock>>>(static_cast<int*>(d_buf.get()), N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy results back to host */
    int h_buf[N];
    CUDA_CHECK(cudaMemcpy(h_buf, d_buf.get(), bytes, cudaMemcpyDeviceToHost));

    /* Print results */
    std::cout << "Device buffer contents:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << h_buf[i] << " ";
    }
    std::cout << std::endl;

    /* d_buf will be automatically freed when it goes out of scope */
    return 0;
}
```