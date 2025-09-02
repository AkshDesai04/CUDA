```cuda
/* 
Aim of the program:
Write a program that deliberately leaks memory on the device by allocating it but never calling `cudaFree`. Run the program multiple times and use the `nvidia-smi` command-line tool to observe how the GPU's memory usage increases.

Thinking:
- In CUDA, device memory is allocated with `cudaMalloc` and freed with `cudaFree`. 
- If we never call `cudaFree`, the allocated memory remains reserved for the lifetime of the process. 
- To observe the leak, the program will continuously allocate a fixed chunk of memory (e.g., 100 MB) in an infinite loop, pausing briefly between allocations. 
- Each iteration will print the total amount of memory allocated so far, allowing the user to watch the growth in GPU memory usage via `nvidia-smi`. 
- The loop terminates when `cudaMalloc` fails (likely due to lack of free device memory). 
- This simple program demonstrates the consequences of not freeing GPU memory and how to monitor it externally. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <unistd.h> // for sleep

int main(int argc, char *argv[]) {
    // Size of each allocation (bytes). Default to 100 MB.
    size_t allocSize = 100 * 1024 * 1024;
    // Allow overriding via command-line argument (optional).
    if (argc > 1) {
        long long requested = atoll(argv[1]); // in MB
        if (requested > 0) {
            allocSize = requested * 1024 * 1024;
        }
    }

    size_t totalAllocated = 0;
    printf("Starting GPU memory leak demo. Each allocation: %zu MB.\n", allocSize / (1024 * 1024));
    printf("Use `nvidia-smi` to monitor GPU memory usage.\n\n");

    while (1) {
        void *devPtr = NULL;
        cudaError_t err = cudaMalloc(&devPtr, allocSize);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed after allocating %zu MB total: %s\n",
                    totalAllocated / (1024 * 1024), cudaGetErrorString(err));
            break;
        }
        totalAllocated += allocSize;
        printf("Allocated %zu MB. Total allocated so far: %zu MB.\n",
               allocSize / (1024 * 1024), totalAllocated / (1024 * 1024));
        // Intentionally not freeing devPtr to create a memory leak.

        // Sleep for a short while to make the leak visible over time.
        sleep(1);
    }

    // In practice, the program will terminate when cudaMalloc fails.
    // Since we never free, any remaining allocations are released automatically
    // when the process exits, but the leak is demonstrated while the program
    // is running.

    return 0;
}
```