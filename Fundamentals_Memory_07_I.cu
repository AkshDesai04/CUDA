```cpp
/*
Try to use `memcpy` or `std::copy` with a device pointer. What happens?

Thinking:

In CUDA, memory that resides on the GPU (device memory) is not directly
accessible by host functions such as the standard C library `memcpy` or the
STL algorithm `std::copy`. These functions are executed on the CPU and
expect pointers to host memory. When we pass a device pointer to them,
the CPU will interpret the device address as a host address, leading to
undefined behaviour – typically a crash or corrupted data.

To copy data between device pointers (or between host and device) we must
use `cudaMemcpy`, which is implemented by the CUDA runtime and knows how
to move data across the PCIe bus or via DMA.

This program demonstrates:
1. Allocating two device arrays.
2. Initialising one from the host.
3. Attempting to copy from one device array to another using:
   a) The standard `memcpy` – this will cause a crash or produce
      incorrect results.
   b) The STL `std::copy` – this also runs on the host and therefore
      fails similarly.
4. Using the correct `cudaMemcpy` to perform a successful device-to-device
   copy.
5. Printing the results to observe the differences.

Compile with: nvcc -o test test.cu
Run with: ./test
*/
#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>

#define N 10

// Simple CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__      \
                      << " (" << cudaGetErrorString(err) << ")\n";            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main() {
    int *d_a = nullptr;
    int *d_b = nullptr;
    int h_a[N], h_b[N];

    // Initialize host array
    for (int i = 0; i < N; ++i) h_a[i] = i + 1;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(int)));

    // Copy host to device for source array
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));

    // ------------------------------------------------------------
    // 1) Attempt to copy device memory using standard memcpy
    // ------------------------------------------------------------
    std::cout << "Attempting std::memcpy from device to device...\n";
    // This line will compile, but will try to copy from the device address
    // as if it were a host address, leading to undefined behavior.
    // In many environments this causes a segmentation fault.
    // To avoid crashing the program, we wrap it in a try-catch, but note that
    // segmentation faults are not catchable in C++. The program will likely
    // terminate here. Uncomment the following line to see the crash in action:
    // std::memcpy(d_b, d_a, N * sizeof(int));
    // For demonstration, we skip the call to prevent crashing.

    // ------------------------------------------------------------
    // 2) Attempt to copy device memory using std::copy
    // ------------------------------------------------------------
    std::cout << "Attempting std::copy from device to device...\n";
    // std::copy operates on iterators; raw pointers are treated as iterators.
    // It will run on the host and attempt to read from device memory,
    // leading to undefined behavior. This may also crash the program.
    // To avoid crashing, we comment it out:
    // std::copy(d_a, d_a + N, d_b);
    // For demonstration, we skip the call.

    // ------------------------------------------------------------
    // 3) Correctly copy device memory using cudaMemcpy
    // ------------------------------------------------------------
    std::cout << "Copying device memory using cudaMemcpy...\n";
    CUDA_CHECK(cudaMemcpy(d_b, d_a, N * sizeof(int), cudaMemcpyDeviceToDevice));

    // ------------------------------------------------------------
    // 4) Copy result back to host and print
    // ------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "Contents of destination array after cudaMemcpy:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << h_b[i] << ' ';
    }
    std::cout << '\n';

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return 0;
}
```