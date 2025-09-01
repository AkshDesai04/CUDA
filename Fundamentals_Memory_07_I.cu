```cpp
/*
Aim of the program:
Try to use `memcpy` or `std::copy` with a device pointer. What happens?

My thinking:
- In CUDA programming, device memory is allocated on the GPU and is not directly
  addressable from host code. Functions like `std::copy` (which operates on
  iterators/pointers in host address space) and `memcpy` (a C library function
  that expects host pointers) will interpret a device pointer as a normal
  CPU pointer.
- Passing a device pointer to either of these functions compiles fine because
  the compiler cannot distinguish between host and device memory at the type
  level. However, at runtime, the host will attempt to dereference the device
  pointer, which is not in the host's address space. This leads to undefined
  behaviour – most often a segmentation fault or copying garbage values.
- The correct way to transfer data between host and device is to use the CUDA
  API `cudaMemcpy`, which is aware of the memory spaces and performs the
  transfer over PCIe or other interconnects.

The code below demonstrates:
  1. An attempt to use `std::copy` with a device pointer (undefined behaviour).
  2. An attempt to use `memcpy` with a device pointer (also undefined behaviour).
  3. The correct `cudaMemcpy` usage.
  4. Checking the results by copying the device data back to host with
     `cudaMemcpy` and printing it.

Compile with:
  nvcc -std=c++11 -arch=sm_52 copy_device_pointer_demo.cu -o copy_device_pointer_demo

Run:
  ./copy_device_pointer_demo

You will see that the first two copy attempts either crash or produce
nonsense output, while the `cudaMemcpy` approach works correctly.
*/

#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

int main() {
    const int N = 10;
    const size_t size = N * sizeof(int);

    // Host array
    int* h_src = new int[N];
    for (int i = 0; i < N; ++i) h_src[i] = i + 1; // 1, 2, ..., N

    // Device array
    int* d_ptr = nullptr;
    cudaError_t err = cudaMalloc(&d_ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        delete[] h_src;
        return -1;
    }

    // 1. Attempt to use std::copy with device pointer (undefined behavior)
    std::cout << "Attempting std::copy from host to device pointer (undefined behavior)...\n";
    try {
        // This will compile but is UB; may crash or silently copy garbage
        std::copy(h_src, h_src + N, d_ptr);
        std::cout << "std::copy executed (no immediate crash).\n";
    } catch (...) {
        std::cout << "std::copy threw an exception.\n";
    }

    // 2. Attempt to use memcpy with device pointer (undefined behavior)
    std::cout << "Attempting memcpy to device pointer (undefined behavior)...\n";
    // Note: memcpy returns void; cannot catch errors. This may cause a crash.
    memcpy(d_ptr, h_src, size);
    std::cout << "memcpy executed (no immediate crash).\n";

    // 3. Correct method: cudaMemcpy
    std::cout << "Copying with cudaMemcpy (correct method)...\n";
    err = cudaMemcpy(d_ptr, h_src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "cudaMemcpy succeeded.\n";
    }

    // 4. Verify by copying back to host
    int* h_dst = new int[N];
    err = cudaMemcpy(h_dst, d_ptr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy (DeviceToHost) failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Data copied back from device to host:\n";
        for (int i = 0; i < N; ++i) {
            std::cout << h_dst[i] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    delete[] h_src;
    delete[] h_dst;
    cudaFree(d_ptr);

    return 0;
}
```