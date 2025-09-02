/*
Aim: Write a program that enumerates all GPUs on the system, and for each one, prints its name and memory capacity. Then, explicitly set the active device to GPU 1 (if it exists) using cudaSetDevice and run a simple kernel on it.

Thinking:
- We need to query the number of CUDA-capable devices with cudaGetDeviceCount.
- For each device, retrieve cudaDeviceProp via cudaGetDeviceProperties and print the name and totalGlobalMem (in GB for readability).
- If there is at least two devices (index 1 exists), set the active device to 1 with cudaSetDevice(1). Handle the case where there is no GPU 1 gracefully.
- Allocate a small array on the selected device, launch a trivial kernel that writes indices into the array, copy it back, and print a few values to confirm execution.
- Wrap CUDA API calls in an error-checking macro to simplify debugging.
- Use standard C++ iostream for output. The kernel is trivial and uses one block with enough threads to cover the array size.
- Ensure all CUDA API calls are checked for errors, and the program cleans up device memory before exit.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

// Macro for checking CUDA API calls
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__\
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                         \
        }                                                                \
    } while (0)

// Simple kernel that writes its global thread index into the array
__global__ void simpleKernel(int *d_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_array[idx] = idx;
    }
}

int main() {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    std::cout << "Detected " << deviceCount << " CUDA-capable device(s)." << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  Total Global Memory: "
                  << static_cast<double>(prop.totalGlobalMem) / (1ULL << 30)
                  << " GB" << std::endl;
    }

    // If GPU 1 exists, set it as the active device
    if (deviceCount > 1) {
        std::cout << "Setting active device to GPU 1." << std::endl;
        CHECK_CUDA(cudaSetDevice(1));
    } else {
        std::cout << "GPU 1 does not exist. Using default device 0." << std::endl;
        CHECK_CUDA(cudaSetDevice(0));
    }

    // Allocate memory on the selected device
    const int N = 16;
    int *d_array = nullptr;
    CHECK_CUDA(cudaMalloc(&d_array, N * sizeof(int)));

    // Launch the kernel
    int threadsPerBlock = 8;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    simpleKernel<<<blocks, threadsPerBlock>>>(d_array, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy results back to host and print
    int h_array[N];
    CHECK_CUDA(cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "Kernel output (first " << N << " elements):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "  h_array[" << i << "] = " << h_array[i] << std::endl;
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_array));

    std::cout << "Program completed successfully." << std::endl;
    return 0;
}
