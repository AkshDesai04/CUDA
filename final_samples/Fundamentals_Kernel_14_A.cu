```cpp
/*
Trigger `cudaErrorInvalidConfiguration` by requesting more shared memory per block than is available on the device.

Thinking:
- The goal is to produce the specific CUDA error `cudaErrorInvalidConfiguration`, which occurs when a kernel launch configuration exceeds device limits.
- One common limit is the amount of dynamic shared memory per block. Each device has a property `sharedMemPerBlock` that specifies the maximum dynamic shared memory that can be requested during a launch.
- To trigger the error, we query the device's maximum shared memory per block, then attempt to launch a kernel requesting more than that amount.
- We'll create a simple dummy kernel that declares an external shared memory array (`extern __shared__ int sdata[];`). The kernel itself does nothing.
- After launching the kernel with an oversized shared memory request, we call `cudaGetLastError()` to capture the launch error. This function returns an error code if the launch failed.
- We print the error string using `cudaGetErrorString` so the user can see the exact error (`cudaErrorInvalidConfiguration`).
- The program is self‑contained, written in CUDA C, and will compile to a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Dummy kernel that uses dynamic shared memory
__global__ void dummyKernel()
{
    extern __shared__ int sdata[];
    // No operations needed; just declare the array
}

int main()
{
    // Get device properties
    cudaDeviceProp prop;
    int device = 0; // default to device 0
    if (cudaGetDevice(&device) != cudaSuccess) {
        fprintf(stderr, "Failed to get current device.\n");
        return 1;
    }

    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties.\n");
        return 1;
    }

    size_t maxSharedMem = prop.sharedMemPerBlock;
    printf("Device %d (%s) supports %zu bytes of shared memory per block.\n",
           device, prop.name, maxSharedMem);

    // Intentionally request more shared memory than available
    size_t requestedSharedMem = maxSharedMem + 1024; // add 1KB extra
    printf("Attempting to launch kernel requesting %zu bytes of shared memory per block.\n",
           requestedSharedMem);

    // Launch kernel with 1 block and 1 thread, requesting dynamic shared memory
    dummyKernel<<<1, 1, requestedSharedMem>>>();
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s (%d)\n", cudaGetErrorString(err), err);
    } else {
        // If launch succeeded, wait for kernel completion and check for any errors
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel execution failed: %s (%d)\n", cudaGetErrorString(err), err);
        } else {
            printf("Kernel launched and executed successfully (unexpected).\n");
        }
    }

    return 0;
}
```