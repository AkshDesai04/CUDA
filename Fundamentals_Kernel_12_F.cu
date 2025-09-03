```cpp
/* 
Aim: Try to create a `dim3` block variable with a size larger than the device's `maxThreadsPerBlock` limit (e.g., `dim3 block(2048)`). What happens at kernel launch?

Thinking:
- Query the device properties using `cudaGetDeviceProperties` to get `maxThreadsPerBlock`.
- Define a block size that intentionally exceeds this limit, e.g., `dim3 block(2048)`.
- Launch a simple kernel (does nothing but is valid) with this oversized block.
- After the kernel launch, call `cudaGetLastError()` to check if the launch succeeded.
- The kernel launch should fail with `cudaErrorInvalidConfiguration`, because CUDA runtime checks the block size against the device capability.
- We print the error message to confirm this behavior.

This program demonstrates the runtime error that occurs when a kernel launch is attempted with an invalid block configuration.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that does nothing
__global__ void dummyKernel() {
    // No-op
}

int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Device %d: maxThreadsPerBlock = %d\n", device, prop.maxThreadsPerBlock);

    // Intentionally exceed maxThreadsPerBlock
    dim3 block(2048);          // Example oversized block
    dim3 grid(1);

    printf("Launching kernel with block size (%d, %d, %d)...\n",
           block.x, block.y, block.z);

    dummyKernel<<<grid, block>>>();

    // Check for launch error
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel launch succeeded.\n");
    }

    // Optional: synchronize to catch any execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}
```