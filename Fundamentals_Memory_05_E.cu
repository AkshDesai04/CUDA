```cuda
/*
Use `nvidia-smi` to watch the GPU memory usage while your program runs. Step through with a debugger to see the memory increase after `cudaMalloc` and decrease after `cudaFree`.

[Thinking]
The goal is to provide a minimal CUDA program that explicitly allocates a block of GPU memory using `cudaMalloc` and then frees it with `cudaFree`. By running the program and observing the GPU memory usage with `nvidia-smi`, one should see a spike in allocated memory after the `cudaMalloc` call and a drop after the `cudaFree` call. The program includes basic error checking and prompts the user to press Enter before freeing the memory, giving time to observe the memory state. This simple example is ideal for stepping through in a debugger, watching the allocation and deallocation, and correlating the changes with `nvidia-smi` output.

Implementation details:
- Allocate 100 MB of device memory to make the memory change noticeable.
- Use `cudaGetErrorString` for readable error messages.
- Use `getchar()` to pause execution after allocation so the user can run `nvidia-smi` and inspect the GPU memory usage.
- After the user presses Enter, free the memory and print a confirmation message.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    const size_t size = 100 * 1024 * 1024; // 100 MB
    int *d_ptr = nullptr;

    // Allocate memory on the device
    cudaError_t err = cudaMalloc((void **)&d_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Allocated %zu bytes on GPU.\n", size);

    // Pause so the user can run nvidia-smi and watch memory usage
    printf("Press Enter to free memory and exit.\n");
    getchar();

    // Free the allocated memory
    err = cudaFree(d_ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Freed GPU memory.\n");

    return 0;
}
```