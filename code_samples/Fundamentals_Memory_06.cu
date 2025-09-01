/*
Aim of the program:
Write a program that deliberately leaks memory on the device by allocating it but never calling `cudaFree`. Run the program multiple times and use the `nvidia-smi` command-line tool to observe how the GPU's memory usage increases.

Thinking behind the implementation:
- The goal is to allocate device memory that persists while the program is running, but never free it. This will cause the GPU to retain the allocated memory until the process terminates, at which point CUDA will clean up the context. To observe the memory usage while the program is alive, we keep the main thread sleeping or in a loop after allocation.
- Allocate a reasonably large chunk of memory (e.g., 100 MiB) so that the increase is noticeable. The exact size can be adjusted if needed.
- Use standard CUDA runtime API (`cudaMalloc`, `cudaGetDeviceProperties`) and C++ utilities for sleeping (`std::this_thread::sleep_for`) so that the program can be compiled with `nvcc`.
- We include a simple error‑checking macro to report CUDA errors, but we intentionally skip calling `cudaFree`.
- When this program runs, `nvidia-smi` will show that the GPU memory usage remains high. Running the program again will allocate more memory (since the previous allocation was never freed), and the total GPU memory usage will increase further until the machine is rebooted or the process is killed.
- Note that once the process exits, CUDA will release the memory automatically. The memory leak is therefore only visible while the process is alive.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <thread>

// Simple macro for error checking
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << #call << " at line " << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl;          \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main() {
    // Allocate 100 MiB of device memory
    const size_t allocSize = 100 * 1024 * 1024; // 100 MB
    void* d_ptr = nullptr;
    std::cout << "Allocating " << allocSize << " bytes on the GPU." << std::endl;
    CHECK_CUDA(cudaMalloc(&d_ptr, allocSize));

    // Optionally, we could fill the memory to ensure allocation actually takes place
    // but cudaMalloc is sufficient for this demonstration.

    // Get device properties for information
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;

    // Inform user that memory is allocated and the program is now sleeping
    std::cout << "Memory allocated and not freed. Program will sleep for 60 seconds." << std::endl;

    // Sleep for a while to let the user observe memory usage with nvidia-smi
    std::this_thread::sleep_for(std::chrono::seconds(60));

    // Program ends here; CUDA context will be destroyed and memory freed automatically
    std::cout << "Exiting program. GPU memory will be freed automatically." << std::endl;
    return 0;
}
