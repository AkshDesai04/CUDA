/*
Add comments explaining that `cudaDeviceReset()` destroys all allocations and contexts on the current device.

My thinking:
The task is to create a CUDA C++ source file (.cu) that includes a clear comment explaining the effect of `cudaDeviceReset()`. The code should be minimal but functional: allocate some device memory, then call `cudaDeviceReset()`, and finally print a message confirming that the reset was performed. Proper error checking is added to make the example robust. The comment at the top must literally contain the phrase from the prompt, and I also include an explanation of why `cudaDeviceReset()` is useful (it cleans up all resources and contexts). This file can be compiled with nvcc and run on a CUDA-capable device. It demonstrates the cleanup behavior without relying on any external libraries or additional files. 
*/

#include <cuda_runtime.h>
#include <stdio.h>

// Utility macro for error checking
#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

int main() {
    // Allocate a small array on the device
    int *devPtr = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&devPtr, 10 * sizeof(int)));

    // (Optional) Perform some operation here. For this example, we just reset the device.

    // Reset the device.
    // NOTE: cudaDeviceReset() destroys all allocations and contexts on the current device.
    // This means that any memory allocations (e.g., devPtr) and any CUDA context
    // associated with this device are completely cleaned up. After this call,
    // the device is left in a clean state, ready for new contexts and allocations.
    CHECK_CUDA(cudaDeviceReset());

    printf("Device reset completed. All allocations and contexts have been destroyed.\n");
    return 0;
}
