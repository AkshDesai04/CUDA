```cpp
/* 
Aim of the program: 
Try to create a `dim3` block variable with a size larger than the device's `maxThreadsPerBlock` limit (e.g., `dim3 block(2048)`). What happens at kernel launch?

Thinking:
- CUDA imposes a limit on the maximum number of threads per block (`maxThreadsPerBlock`) which depends on the GPU architecture.
- Creating a `dim3` object with a value that exceeds this limit (e.g., `dim3 block(2048)`) does not immediately cause a compile‑time or runtime error; the compiler accepts the value.
- However, when the kernel is launched with such a block configuration, the CUDA runtime will check the configuration against the device limits.
- If the requested number of threads per block exceeds `maxThreadsPerBlock`, the launch fails and `cudaGetLastError()` will return `cudaErrorLaunchOutOfResources` (or `cudaErrorInvalidConfiguration` on older drivers).
- The program demonstrates this by querying the device's `maxThreadsPerBlock`, attempting a kernel launch with a too‑large block, and printing the resulting CUDA error.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel()
{
    // Simple kernel that does nothing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Just prevent unused warning
    if (idx == 0) {
        printf("Kernel running\n");
    }
}

int main()
{
    // Get device properties
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device: %s\n", prop.name);
    printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);

    // Intentionally set block size larger than the limit
    dim3 block(2048);   // For many GPUs, 2048 > maxThreadsPerBlock
    dim3 grid(1);

    // Launch the kernel
    dummyKernel<<<grid, block>>>();

    // Synchronize to catch any launch errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel launched successfully (unexpected).\n");
    }

    // Alternatively, check error immediately after launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError after launch: %s\n", cudaGetErrorString(err));
    }

    return 0;
}
```