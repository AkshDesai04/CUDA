/*
Allocate memory and use `cudaMemsetAsync` with a non-default stream.

Thinking process:
1. We need to allocate device memory using cudaMalloc.
2. Create a non-default CUDA stream using cudaStreamCreate.
3. Use cudaMemsetAsync to set the memory asynchronously on that stream.
4. After setting, we copy the memory back to host using cudaMemcpyAsync with the same stream.
5. Synchronize the stream with cudaStreamSynchronize to ensure all operations finish before we read from host memory.
6. Validate the data on the host (e.g., print the first element).
7. Clean up: free device memory, destroy stream, and exit.

The program will:
- Allocate an array of 1024 integers on the device.
- Asynchronously set each integer to 0xFFFFFFFF using cudaMemsetAsync.
- Copy the array back to host.
- Print the first element to verify that the set operation succeeded.
- Use proper error checking for CUDA calls.
*/

#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err) << std::endl;     \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)

int main() {
    const size_t N = 1024;                     // number of integers
    int* dev_arr = nullptr;
    int* host_arr = new int[N];

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&dev_arr, N * sizeof(int)));

    // Create a non-default stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Asynchronously set all integers to 0xFFFFFFFF on the device
    const unsigned char value = 0xFF;          // byte value to set
    CHECK_CUDA(cudaMemsetAsync(dev_arr, value, N * sizeof(int), stream));

    // Asynchronously copy the device array back to host
    CHECK_CUDA(cudaMemcpyAsync(host_arr, dev_arr, N * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));

    // Wait for all operations in the stream to complete
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Verify the first element
    std::cout << "First element after cudaMemsetAsync: 0x"
              << std::hex << host_arr[0] << std::dec << std::endl;

    // Clean up
    delete[] host_arr;
    CHECK_CUDA(cudaFree(dev_arr));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return 0;
}
