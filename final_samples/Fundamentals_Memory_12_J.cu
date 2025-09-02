```cpp
/*
Aim of the program: 
Create a similar RAII wrapper for pinned host memory (`cudaMallocHost`/`cudaFreeHost`).

Thinking:
1. Pinned host memory (page-locked) is used to accelerate host‑to‑device and device‑to‑host transfers. 
   Allocation is done with `cudaMallocHost(void **ptr, size_t size)` and freed with `cudaFreeHost(void *ptr)`.  
2. An RAII wrapper should:
   - Allocate in the constructor and free in the destructor.
   - Disable copying (to avoid double‑free) but support move semantics.
   - Provide convenient accessors: raw pointer, size, and maybe a method to get a `cudaStream_t` ready transfer.
   - Throw an exception if allocation fails.
3. The wrapper will be a template `PinnedHostMemory<T>` so it can be used for any element type, similar to the 
   device memory RAII wrapper pattern.  
4. Provide a small `main` demonstrating allocation, initialization, and transfer to device memory using 
   `cudaMemcpyAsync`.  The example will also show that the destructor frees the pinned memory automatically. 
5. Error handling: a helper macro `CUDA_CHECK` that checks return codes of CUDA runtime API calls and throws 
   `std::runtime_error` with a descriptive message.  
6. All code is in a single `.cu` file as requested. No external dependencies other than the CUDA runtime. 
7. We keep the code simple but idiomatic: `noexcept` where appropriate, use `std::move` for move constructor/assignment.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

/* Helper macro for error checking */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            throw std::runtime_error(std::string("CUDA error: ") +           \
                                     cudaGetErrorString(err) + " (" +        \
                                     std::to_string(__LINE__) + ") in " +  \
                                     __FILE__);                              \
        }                                                                    \
    } while (0)

/* RAII wrapper for pinned host memory */
template <typename T>
class PinnedHostMemory {
public:
    /* Construct with a given number of elements */
    explicit PinnedHostMemory(std::size_t count)
        : ptr_(nullptr), count_(count) {
        if (count_ == 0) {
            return; // no allocation needed for zero size
        }
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
    }

    /* Destructor frees the pinned memory */
    ~PinnedHostMemory() noexcept {
        if (ptr_) {
            cudaFreeHost(ptr_);
        }
    }

    /* Disable copy semantics */
    PinnedHostMemory(const PinnedHostMemory&) = delete;
    PinnedHostMemory& operator=(const PinnedHostMemory&) = delete;

    /* Enable move semantics */
    PinnedHostMemory(PinnedHostMemory&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    PinnedHostMemory& operator=(PinnedHostMemory&& other) noexcept {
        if (this != &other) {
            // free existing memory
            if (ptr_) {
                cudaFreeHost(ptr_);
            }
            // move
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    /* Accessors */
    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }
    std::size_t byte_size() const noexcept { return count_ * sizeof(T); }

private:
    T* ptr_;
    std::size_t count_;
};

/* Simple demonstration program */
int main() {
    try {
        const std::size_t N = 1024;
        /* Allocate pinned host memory */
        PinnedHostMemory<float> h_data(N);
        /* Initialize data */
        for (std::size_t i = 0; i < N; ++i) {
            h_data.data()[i] = static_cast<float>(i);
        }

        /* Allocate device memory */
        float* d_data = nullptr;
        CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));

        /* Create a stream for async copy */
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        /* Async copy from pinned host to device */
        CUDA_CHECK(cudaMemcpyAsync(d_data, h_data.data(), N * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));

        /* Synchronize stream to ensure copy is complete */
        CUDA_CHECK(cudaStreamSynchronize(stream));

        /* Verify a few values on device (by copying back) */
        float h_verify[10];
        CUDA_CHECK(cudaMemcpy(h_verify, d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "First 10 values copied to device: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << h_verify[i] << ' ';
        }
        std::cout << std::endl;

        /* Clean up device memory and stream */
        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaStreamDestroy(stream));

        /* h_data will be automatically freed when going out of scope */
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```