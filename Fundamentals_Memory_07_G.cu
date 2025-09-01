```cpp
/* 
Aim of the program: 
Write a __global__ function that takes an int* and dereferences it. Call it with a host pointer. Observe the error at runtime. 

Thinking: 
To demonstrate the runtime error that occurs when a kernel is launched with a host pointer, 
we will create a simple CUDA program that defines a __global__ kernel which accepts an 
int* argument and dereferences it (e.g., int val = *ptr;). We will allocate an integer 
on the host (CPU) and pass its address directly to the kernel launch. Since device 
kernels can only dereference pointers that refer to device memory (or unified memory), 
passing a host pointer will cause the CUDA runtime to detect an invalid device 
pointer during the kernel launch. The program will check the error returned by 
cudaGetLastError() and also after cudaDeviceSynchronize(), printing out the 
error code and message. This will allow us to observe the runtime error in a 
controlled manner. No device memory allocation is performed; the error is purely
a consequence of the invalid pointer being passed to the kernel.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(err)                                              \
    if (err != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error: %s (err_num=%d)\n",             \
                cudaGetErrorString(err), err);                      \
        exit(EXIT_FAILURE);                                          \
    }

/* __global__ kernel that dereferences an int pointer */
__global__ void derefKernel(int *ptr)
{
    // Attempt to read the integer value
    int val = *ptr;
    // Optionally, print the value (requires compute capability >= 2.0)
    printf("Kernel read value: %d\n", val);
}

int main()
{
    int h_val = 42;                      // Host integer

    // Launch kernel with host pointer (intentionally incorrect)
    derefKernel<<<1, 1>>>(&h_val);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("Kernel launched successfully (unexpected)\n");
    }

    // Synchronize to catch any runtime errors that happen during execution
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Runtime error during kernel execution: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("Kernel executed successfully (unexpected)\n");
    }

    return 0;
}
```