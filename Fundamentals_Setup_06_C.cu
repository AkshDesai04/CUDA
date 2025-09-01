```cuda
/*
Write a program that checks if a hypothetical block configuration of (32, 32, 2) would be valid.

Thinking:
1. The aim is to determine whether the block configuration (32, 32, 2) satisfies CUDA hardware limits.
2. Key limits to check:
   - The product of the dimensions (threads per block) must not exceed the device's maximum threads per block.
   - Each individual dimension must not exceed the device's maximum for that dimension.
3. Retrieve these limits via cudaGetDeviceProperties.
4. Compute threads per block for (32,32,2) = 32 * 32 * 2 = 2048.
5. Compare against:
   - maxThreadsPerBlock
   - maxThreadsDim[0] (x), maxThreadsDim[1] (y), maxThreadsDim[2] (z)
6. Output whether each check passes and whether the overall configuration is valid.
7. Since typical GPUs limit maxThreadsPerBlock to 1024 (or less for older compute capabilities),
   the configuration should be reported as invalid due to exceeding the thread count.
8. The program will be a self-contained .cu file, compilable with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Hypothetical block configuration
    int blockDimX = 32;
    int blockDimY = 32;
    int blockDimZ = 2;

    int threadsPerBlock = blockDimX * blockDimY * blockDimZ;

    printf("Device: %s\n", prop.name);
    printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Maximum block dimension X: %d\n", prop.maxThreadsDim[0]);
    printf("Maximum block dimension Y: %d\n", prop.maxThreadsDim[1]);
    printf("Maximum block dimension Z: %d\n", prop.maxThreadsDim[2]);
    printf("\n");

    printf("Checking block configuration (%d, %d, %d):\n", blockDimX, blockDimY, blockDimZ);
    printf("Threads per block: %d\n", threadsPerBlock);

    bool valid = true;

    if (blockDimX > prop.maxThreadsDim[0]) {
        printf("  ❌ X dimension exceeds maximum (%d > %d)\n", blockDimX, prop.maxThreadsDim[0]);
        valid = false;
    } else {
        printf("  ✅ X dimension is within limit (%d <= %d)\n", blockDimX, prop.maxThreadsDim[0]);
    }

    if (blockDimY > prop.maxThreadsDim[1]) {
        printf("  ❌ Y dimension exceeds maximum (%d > %d)\n", blockDimY, prop.maxThreadsDim[1]);
        valid = false;
    } else {
        printf("  ✅ Y dimension is within limit (%d <= %d)\n", blockDimY, prop.maxThreadsDim[1]);
    }

    if (blockDimZ > prop.maxThreadsDim[2]) {
        printf("  ❌ Z dimension exceeds maximum (%d > %d)\n", blockDimZ, prop.maxThreadsDim[2]);
        valid = false;
    } else {
        printf("  ✅ Z dimension is within limit (%d <= %d)\n", blockDimZ, prop.maxThreadsDim[2]);
    }

    if (threadsPerBlock > prop.maxThreadsPerBlock) {
        printf("  ❌ Total threads per block exceed maximum (%d > %d)\n", threadsPerBlock, prop.maxThreadsPerBlock);
        valid = false;
    } else {
        printf("  ✅ Total threads per block within limit (%d <= %d)\n", threadsPerBlock, prop.maxThreadsPerBlock);
    }

    printf("\nOverall configuration validity: %s\n", valid ? "VALID" : "INVALID");

    return 0;
}
```