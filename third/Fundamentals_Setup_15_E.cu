/*
Aim: Use `nvidia-smi` to monitor GPU memory. Run a program that allocates memory but doesn't free it. Then run a program that allocates memory and then calls `cudaDeviceReset()`. Observe how `nvidia-smi` shows the memory being reclaimed in the second case.

Thinking:
1. The user wants to see how GPU memory is reported by `nvidia-smi` when a CUDA program allocates memory and does not free it, versus when it calls `cudaDeviceReset()`.  
2. We can create a single CUDA program that accepts a command‑line argument to choose between the two behaviors:  
   - `"noreset"` (default): allocate memory, keep it allocated until the program exits.  
   - `"reset"`: allocate memory, then call `cudaDeviceReset()` which frees all allocations and resets the device.  
3. To make the memory usage observable, we should:  
   - Allocate a reasonably large amount of memory (e.g., 512 MiB).  
   - Query and print the free/total memory before allocation, after allocation, and after reset (if applicable).  
   - Sleep for a short period (e.g., 10 seconds) after allocation so the user can run `nvidia-smi` and see the reported usage.  
4. Since `cudaDeviceReset()` will terminate the program, the sleep should occur before the reset; the program will then exit cleanly after the reset.  
5. The code will include necessary error checks, and use the C interface (`cuda.h`) to keep it minimal and portable.  
6. Compile with `nvcc -o mem_monitor mem_monitor.cu` and run as:  
   - `./mem_monitor` or `./mem_monitor noreset` to see memory not reclaimed.  
   - `./mem_monitor reset` to see memory reclaimed by `cudaDeviceReset()`.  
7. The multi-line comment at the top of the file contains the aim and this full reasoning, as requested. The rest of the file is plain CUDA C code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>

#define MEMORY_SIZE (512 * 1024 * 1024)  // 512 MiB

void print_mem_info(const char *msg) {
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("%s - Free: %zu MiB, Total: %zu MiB\n",
           msg, free_mem / (1024 * 1024), total_mem / (1024 * 1024));
}

int main(int argc, char **argv) {
    const char *mode = (argc > 1) ? argv[1] : "noreset";
    printf("Running in mode: %s\n", mode);

    print_mem_info("Before allocation");

    // Allocate memory
    void *devPtr = NULL;
    cudaError_t err = cudaMalloc(&devPtr, MEMORY_SIZE);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    printf("Allocated %zu MiB on device.\n", MEMORY_SIZE / (1024 * 1024));

    print_mem_info("After allocation");

    // Sleep to allow user to observe nvidia-smi
    printf("Sleeping for 10 seconds to allow nvidia-smi observation...\n");
    sleep(10);

    if (strcmp(mode, "reset") == 0) {
        printf("Calling cudaDeviceReset() to free memory.\n");
        err = cudaDeviceReset();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(err));
            return EXIT_FAILURE;
        }
        print_mem_info("After cudaDeviceReset");
    } else {
        printf("Not resetting device; memory remains allocated until program exit.\n");
    }

    // If we didn't call cudaDeviceReset, free manually (optional)
    if (strcmp(mode, "noreset") == 0) {
        err = cudaFree(devPtr);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
            return EXIT_FAILURE;
        }
        printf("Manually freed allocated memory.\n");
    }

    printf("Program exiting.\n");
    return EXIT_SUCCESS;
}
