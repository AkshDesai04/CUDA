/*
Trigger `cudaErrorInvalidConfiguration` by requesting more shared memory per block than is available on the device.
Thought process:
- The goal is to produce a runtime error on kernel launch by specifying a shared memory size that exceeds the device's per-block limit.
- CUDA kernels can request dynamic shared memory by specifying the third argument to the launch configuration (shared memory size in bytes).
- Each device reports the maximum allowed shared memory per block via `cudaDeviceProp::sharedMemPerBlock`.
- If we request more than that value, the runtime should return `cudaErrorInvalidConfiguration`.
- The program will query the device properties, compute an over‑requested shared memory size, attempt a kernel launch, and then check the returned error code.
- The kernel itself can be empty because we only care about the launch configuration error.
- After the launch we call `cudaGetLastError()` to retrieve the actual error code, and print a message indicating success or failure.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void dummyKernel(void)
{
    // Empty kernel just to trigger launch configuration
}

int main(void)
{
    cudaDeviceProp prop;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    size_t maxSharedMem = prop.sharedMemPerBlock; // maximum allowed shared memory per block

    // Request more shared memory than the device allows
    size_t requestedSharedMem = maxSharedMem + 1024; // add 1KB to exceed limit

    printf("Device %d properties:\n", dev);
    printf("  Name: %s\n", prop.name);
    printf("  Max shared memory per block: %zu bytes\n", maxSharedMem);
    printf("  Requesting %zu bytes of shared memory per block.\n", requestedSharedMem);

    // Launch kernel with over‑requested shared memory
    dummyKernel<<<1, 1, requestedSharedMem>>>(0);
    cudaError_t err = cudaGetLastError();

    if (err == cudaSuccess)
    {
        printf("Kernel launch succeeded (unexpected).\n");
    }
    else if (err == cudaErrorInvalidConfiguration)
    {
        printf("Kernel launch failed with cudaErrorInvalidConfiguration as expected.\n");
    }
    else
    {
        printf("Kernel launch failed with unexpected error: %s\n", cudaGetErrorString(err));
    }

    // Clean up
    cudaDeviceReset();
    return 0;
}
