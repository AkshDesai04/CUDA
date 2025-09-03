```cuda
/*
 * Aim of the program: 
 *  Try to use memcpy or std::copy with a device pointer. What happens?
 *
 * Thinking:
 *  In CUDA C, device memory is allocated on the GPU via cudaMalloc.
 *  The CPU cannot directly dereference pointers that refer to GPU memory.
 *  If we attempt to use standard host memory copy functions such as
 *  memcpy() or the STL algorithm std::copy() with a device pointer,
 *  the compiler will happily accept the code (since the pointer type is
 *  just a void* or int*), but at runtime the CPU will try to read or
 *  write through that pointer as if it were a normal host address.
 *  This results in undefined behavior: typically a segmentation fault
 *  or the copy will silently do nothing and the GPU memory will remain
 *  uninitialized. The correct way to transfer data between host and
 *  device is to use cudaMemcpy() or related CUDA API calls.
 *
 *  The program below demonstrates:
 *    1. Allocating a host array and filling it with known values.
 *    2. Allocating device memory.
 *    3. Attempting to copy data from host to device using std::copy()
 *       (will compile but cause a crash or incorrect data).
 *    4. Attempting to copy data using memcpy() (same issue).
 *    5. Then performing the correct copy with cudaMemcpy() and verifying
 *       the result by launching a simple kernel that prints the device
 *       array contents.
 *
 *  Because the wrong copies are likely to crash the program, we surround
 *  them with try/catch blocks and error checks, but the primary purpose
 *  is to show that they do not work as intended.
 */

#include <iostream>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA error at " << __FILE__ << ":"        \
                      << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Simple kernel to print device array
__global__ void printArrayKernel(int *dev_arr, int N)
{
    int idx = threadIdx.x;
    if (idx < N) {
        printf("Device[%d] = %d\n", idx, dev_arr[idx]);
    }
}

int main()
{
    const int N = 8;
    const int bytes = N * sizeof(int);

    // Host array
    int h_arr[N];
    for (int i = 0; i < N; ++i) h_arr[i] = i * 10;

    // Device array
    int *d_arr = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_arr, bytes));

    std::cout << "Attempting std::copy from host to device pointer (undefined behavior)...\n";
    try {
        // This will compile but is incorrect.
        std::copy(h_arr, h_arr + N, (int*)d_arr);
        std::cout << "std::copy executed (may have caused crash or undefined behavior).\n";
    } catch (...) {
        std::cerr << "Caught exception during std::copy.\n";
    }

    // Try memcpy
    std::cout << "Attempting memcpy from host to device pointer (undefined behavior)...\n";
    try {
        memcpy(d_arr, h_arr, bytes);
        std::cout << "memcpy executed (may have caused crash or undefined behavior).\n";
    } catch (...) {
        std::cerr << "Caught exception during memcpy.\n";
    }

    // Launch kernel to print device array (likely garbage)
    std::cout << "Printing device array after incorrect copies:\n";
    printArrayKernel<<<1, N>>>(d_arr, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Now perform correct copy
    std::cout << "\nPerforming correct copy using cudaMemcpy...\n";
    CHECK_CUDA(cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice));

    // Verify by printing again
    std::cout << "Printing device array after correct copy:\n";
    printArrayKernel<<<1, N>>>(d_arr, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Clean up
    CHECK_CUDA(cudaFree(d_arr));

    return 0;
}
```