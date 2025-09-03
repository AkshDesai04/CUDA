```cpp
/*
Write a `try_alloc(size_t bytes)` function that returns `true` if `cudaMalloc` succeeds (and then immediately frees it) and `false` otherwise.

Thinking:
- The function must attempt to allocate a buffer of `bytes` on the GPU.
- If `cudaMalloc` returns `cudaSuccess`, we immediately free the allocated memory with `cudaFree`.
- We then return `true` to indicate success.
- If `cudaMalloc` fails, we return `false`.
- We need to include the CUDA runtime header (`cuda_runtime.h`) and standard C++ headers for I/O.
- For demonstration, a simple `main` will call `try_alloc` with a few sizes and print the results.
- Error handling is straightforward: we ignore the actual error code other than success/failure.
- The function is `__host__` because it runs on the CPU.
*/

#include <cuda_runtime.h>
#include <iostream>

/**
 * Attempts to allocate `bytes` of device memory.
 * @param bytes Number of bytes to allocate.
 * @return true if allocation succeeded, false otherwise.
 */
bool try_alloc(size_t bytes) {
    void* d_ptr = nullptr;
    cudaError_t err = cudaMalloc(&d_ptr, bytes);
    if (err == cudaSuccess) {
        // Allocation succeeded; free immediately.
        cudaFree(d_ptr);
        return true;
    }
    // Allocation failed.
    return false;
}

int main() {
    // Example sizes to test the function.
    size_t sizes[] = {1024, 1024 * 1024, 1024ULL * 1024ULL * 1024ULL * 8ULL}; // 8GB
    for (size_t sz : sizes) {
        bool result = try_alloc(sz);
        std::cout << "try_alloc(" << sz << " bytes) returned " << (result ? "true" : "false") << std::endl;
    }
    return 0;
}
```