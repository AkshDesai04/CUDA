```cuda
/*
What happens if you allocate memory but your program crashes before `cudaFree` is called? Does the memory get reclaimed?

Thinking:
- In CUDA, device memory allocation is tied to the process's CUDA context.
- When the process exits normally, the CUDA driver automatically releases all resources, including device memory.
- If the process crashes (for example, via a segmentation fault), the operating system terminates the process.
- The CUDA driver receives a notification that the context is being destroyed and frees all memory associated with it.
- Thus, the memory is reclaimed after the crash; it is not leaked to other processes or persistent.
- This program demonstrates the scenario by allocating 1 MiB of device memory, printing its address, and then deliberately
  causing a host segmentation fault. No `cudaFree` is called. After the crash, you can check with tools like `nvidia-smi`
  that the memory has been reclaimed.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(void)
{
    const size_t size = 1 << 20;  // 1 MiB
    void *devPtr = NULL;

    cudaError_t err = cudaMalloc(&devPtr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Allocated %zu bytes on device. devPtr = %p\n", size, devPtr);

    /* Intentionally crash the host process to simulate a crash before cudaFree */
    int *ptr = NULL;
    *ptr = 42;  // This causes a segmentation fault

    /* The following line would never be executed */
    // cudaFree(devPtr);

    return EXIT_SUCCESS;
}
```