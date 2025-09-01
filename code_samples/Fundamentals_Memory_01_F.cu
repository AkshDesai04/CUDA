/*
Aim of the program:
Try to `cudaMalloc` zero bytes. Does it return an error or a valid pointer?

Thinking process:
- According to the CUDA Runtime API documentation, cudaMalloc should accept a size of 0 and return a non-NULL pointer (implementation-defined). This is because the runtime may return a dummy pointer to represent a zero-sized allocation, and the caller is not supposed to dereference it. In practice, most CUDA drivers return a non-NULL pointer when allocating 0 bytes and return cudaSuccess. However, there is no guarantee across all driver versions or hardware; some drivers might return cudaErrorInvalidValue.
- To verify this, we write a small host program that attempts to allocate 0 bytes, checks the return code, prints whether the allocation succeeded, prints the pointer value, and then frees the pointer if it is not NULL.
- The program uses standard host C++ code with the CUDA Runtime API (`cuda_runtime.h`). It compiles with nvcc and runs on any CUDA-capable device.
- We also print the error string returned by cudaGetErrorString if the allocation fails.

The program will produce output similar to:
  cudaMalloc(0) succeeded. Pointer: 0x7fff8c000000
or
  cudaMalloc(0) failed: cudaErrorInvalidValue

This demonstrates whether the CUDA runtime accepts a zero-byte allocation or not.

*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    void *ptr = nullptr;
    size_t size = 0;

    cudaError_t err = cudaMalloc(&ptr, size);
    std::cout << "cudaMalloc(" << size << ") returned: ";
    if (err == cudaSuccess) {
        std::cout << "cudaSuccess\n";
        std::cout << "Pointer value: " << ptr << "\n";
    } else {
        std::cout << err << " (" << cudaGetErrorString(err) << ")\n";
    }

    // If we got a non-null pointer, attempt to free it.
    if (ptr != nullptr) {
        cudaError_t freeErr = cudaFree(ptr);
        if (freeErr != cudaSuccess) {
            std::cerr << "cudaFree failed: " << freeErr
                      << " (" << cudaGetErrorString(freeErr) << ")\n";
        } else {
            std::cout << "cudaFree succeeded.\n";
        }
    }

    return 0;
}
