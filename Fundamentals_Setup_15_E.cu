/*
Aim: Use `nvidia-smi` to monitor GPU memory. Run a program that allocates memory but doesn't free it. Then run a program that allocates memory and then calls `cudaDeviceReset()`. Observe how `nvidia-smi` shows the memory being reclaimed in the second case.

Thinking: The goal is to demonstrate that GPU memory allocated via `cudaMalloc` remains allocated until the device is reset or the process ends. In the first variant we will allocate a chunk of memory (e.g., 256 MB), keep the program alive for a while, and never free the allocation or reset the device. `nvidia-smi` should report that the GPU memory is still in use during this period. In the second variant we allocate the same amount of memory and then invoke `cudaDeviceReset()`, which cleans up all resources on the device, including our allocation. After the reset the memory usage reported by `nvidia-smi` should drop back to the baseline, even though the process is still running.

Implementation details:
1. Accept a single command‑line argument:
   * `"noreset"` – allocate memory, keep the allocation alive for 30 s, then exit.
   * `"reset"`   – allocate memory, call `cudaDeviceReset()`, then exit.
2. Use a 256 MB allocation (`256 * 1024 * 1024` bytes).  
3. Use a simple macro `CHECK_CUDA` for error handling.  
4. Print status messages so the user can correlate them with `nvidia-smi` output.  
5. Sleep for 30 s (via `std::this_thread::sleep_for`) to allow the user to observe the memory usage with `nvidia-smi`.  
6. Compile with `nvcc -o mem_demo mem_demo.cu`.

The single .cu file below contains all the code. Compile and run it as described above to observe the difference in GPU memory usage between the two scenarios.

*/

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error at " << __FILE__ << ":"            \
                      << __LINE__ << " - " << cudaGetErrorString(err)   \
                      << std::endl;                                    \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Dummy kernel that does nothing but forces the GPU to stay active
__global__ void dummyKernel() {
    // No-op
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <noreset|reset>" << std::endl;
        return EXIT_FAILURE;
    }

    const size_t allocationSize = 256 * 1024 * 1024; // 256 MB
    void* d_ptr = nullptr;

    std::cout << "Allocating " << allocationSize / (1024 * 1024)
              << " MB of GPU memory." << std::endl;
    CHECK_CUDA(cudaMalloc(&d_ptr, allocationSize));

    // Launch dummy kernel to ensure the allocation is touched
    dummyKernel<<<1, 1>>>();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    if (std::strcmp(argv[1], "noreset") == 0) {
        std::cout << "Running in noreset mode. Memory will not be freed." << std::endl;
        std::cout << "You can run 'nvidia-smi' now to observe the memory usage." << std::endl;
        std::cout << "Sleeping for 30 seconds..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(30));
        std::cout << "Exiting without freeing memory." << std::endl;
    } else if (std::strcmp(argv[1], "reset") == 0) {
        std::cout << "Running in reset mode. Calling cudaDeviceReset()." << std::endl;
        // Reset device: this will free all allocations and reset the device state
        CHECK_CUDA(cudaDeviceReset());
        std::cout << "Device reset complete. Memory should be reclaimed." << std::endl;
        std::cout << "You can run 'nvidia-smi' now to verify." << std::endl;
    } else {
        std::cerr << "Invalid argument. Use 'noreset' or 'reset'." << std::endl;
        return EXIT_FAILURE;
    }

    // Allow some time after reset for the user to check nvidia-smi
    std::this_thread::sleep_for(std::chrono::seconds(5));

    return EXIT_SUCCESS;
}
