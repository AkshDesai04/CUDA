/*
Aim: If a kernel requires 16KB of shared memory, calculate the maximum number of threads you can launch in a block if the device limit is 48KB per block and 1024 threads (assuming no other constraints).

Thinking:
- The kernel requires 16KB of shared memory per block, regardless of the number of threads it contains.
- The device has a shared memory limit of 48KB per block. Since 48KB / 16KB = 3, the limit does not impose a stricter restriction on the number of blocks that can run concurrently; it only tells us that at most 3 blocks could be resident if each uses 16KB.
- The maximum number of threads per block is governed by the hardware limit of 1024 threads (obtained from cudaDeviceProp.maxThreadsPerBlock). Because there are no additional constraints such as registers or shared memory per thread, we can launch the full 1024 threads in a block.
- Therefore, the program will query the device properties and compute the maximum threads per block, which in this scenario will be 1024.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Shared memory required by the kernel per block (in bytes)
    const size_t sharedMemRequired = 16 * 1024; // 16KB

    // Device limits
    size_t sharedMemPerBlockLimit = prop.sharedMemPerBlock;
    int   maxThreadsPerBlock      = prop.maxThreadsPerBlock;

    // Compute the maximum number of blocks that can be resident given shared memory requirement
    int maxBlocksBySharedMem = static_cast<int>(sharedMemPerBlockLimit / sharedMemRequired);

    // Since the shared memory requirement does not depend on the number of threads,
    // we can still use the full maximum number of threads per block.
    int maxThreads = maxThreadsPerBlock;

    printf("Device %d properties:\n", device);
    printf("  Shared memory per block: %zu bytes\n", sharedMemPerBlockLimit);
    printf("  Max threads per block: %d\n", maxThreadsPerBlock);
    printf("\n");
    printf("Kernel shared memory requirement per block: %zu bytes\n", sharedMemRequired);
    printf("Maximum number of resident blocks (shared memory limited): %d\n", maxBlocksBySharedMem);
    printf("Maximum number of threads you can launch in a block (assuming no other constraints): %d\n", maxThreads);

    return 0;
}
