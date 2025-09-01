```cuda
/* 
Aim of the program: 
Use `nvidia-smi` to monitor GPU memory. Run a program that allocates memory but doesn't free it. 
Then run a program that allocates memory and then calls `cudaDeviceReset()`. 
Observe how `nvidia-smi` shows the memory being reclaimed in the second case. 

Thinking:
The goal is to demonstrate the difference in GPU memory usage between a program that leaks device memory
and a program that cleans up by resetting the device. 
The CUDA runtime manages device memory allocations internally; if we never free allocated memory, 
it remains allocated until the process terminates. 
Calling `cudaDeviceReset()` releases all resources (including memory) associated with the current device,
effectively cleaning up without having to explicitly free each allocation.

To observe the behavior, the program offers two modes selectable via a command line argument:
- "leak": allocates a large buffer (1 GB of floats) and never frees it, then sleeps for 60 seconds.
- "reset": allocates the same buffer, then calls `cudaDeviceReset()`, which frees the buffer.
Both modes print the device's free/total memory before and after the allocation/cleanup so the
user can compare the numbers while monitoring with `nvidia-smi`.

Implementation details:
- Use `cudaMemGetInfo` to report free and total memory.
- Use a macro `CHECK` for error handling.
- Sleep for 60 s in the leak mode to give the user time to run `nvidia-smi`.
- The program requires CUDA toolkit and can be compiled with `nvcc`.
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cuda_runtime.h>

#define CHECK(call)                                            \
    do {                                                       \
        cudaError_t err = (call);                              \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

void print_mem_info(const char* msg) {
    size_t free_mem, total_mem;
    CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("%s: Free memory: %zu MB, Total memory: %zu MB\n",
           msg,
           free_mem / (1024 * 1024),
           total_mem / (1024 * 1024));
}

void leak_mode() {
    printf("=== Leak Mode ===\n");
    print_mem_info("Before allocation");

    const size_t num_floats = 256 * 1024 * 1024; // 256M floats ~ 1 GB
    float *d_ptr = nullptr;
    CHECK(cudaMalloc(&d_ptr, num_floats * sizeof(float)));

    print_mem_info("After allocation");

    printf("Sleeping for 60 seconds. Use `nvidia-smi` to observe memory usage.\n");
    std::this_thread::sleep_for(std::chrono::seconds(60));

    printf("Exiting without freeing memory. Memory should still be allocated until process exit.\n");
    // Intentionally not freeing d_ptr
}

void reset_mode() {
    printf("=== Reset Mode ===\n");
    print_mem_info("Before allocation");

    const size_t num_floats = 256 * 1024 * 1024; // 256M floats ~ 1 GB
    float *d_ptr = nullptr;
    CHECK(cudaMalloc(&d_ptr, num_floats * sizeof(float)));

    print_mem_info("After allocation");

    printf("Calling cudaDeviceReset() to release all resources.\n");
    CHECK(cudaDeviceReset());

    print_mem_info("After cudaDeviceReset()");

    printf("Program finished. Memory should be reclaimed immediately.\n");
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s [leak|reset]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* mode = argv[1];
    if (strcmp(mode, "leak") == 0) {
        leak_mode();
    } else if (strcmp(mode, "reset") == 0) {
        reset_mode();
    } else {
        fprintf(stderr, "Unknown mode '%s'. Use 'leak' or 'reset'.\n", mode);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```