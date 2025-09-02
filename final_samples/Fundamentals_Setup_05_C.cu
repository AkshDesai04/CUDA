```cpp
/*
If a kernel uses 64 registers per thread and the max threads per block is 1024, calculate the total registers required by a full block. Compare this to the max registers per block to see if such a launch is possible.

Thinking:
- We need to query the device properties to obtain:
  * The maximum number of threads per block (cudaDeviceProp::maxThreadsPerBlock).
  * The maximum number of registers per block (cudaDeviceProp::regsPerBlock).
- The kernel register requirement per thread is given as 64.
- A full block would therefore require: 64 registers/thread * 1024 threads/block = 65536 registers/block.
- We compare this required number against the device's regsPerBlock to determine if the launch configuration is permissible.
- If the required registers are less than or equal to regsPerBlock, the launch is possible; otherwise, it is not.
- For demonstration purposes, we will also launch a trivial kernel that uses the specified registers to confirm that the configuration is supported.
*/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void dummyKernel() {
    // Dummy kernel that does nothing but consumes registers.
    // Each thread will use a local array to occupy registers.
    int reg[64]; // Force 64 registers per thread (approximate)
    // Prevent compiler optimizations
    for (int i = 0; i < 64; ++i) reg[i] = threadIdx.x + i;
}

int main() {
    cudaDeviceProp prop;
    int device = 0;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    const int regsPerThread = 64;
    const int maxThreadsPerBlock = prop.maxThreadsPerBlock; // Typically 1024
    const int requiredRegsPerBlock = regsPerThread * maxThreadsPerBlock;

    printf("Device %d properties:\n", device);
    printf("  Max threads per block          : %d\n", prop.maxThreadsPerBlock);
    printf("  Max registers per block        : %d\n", prop.regsPerBlock);
    printf("\nKernel configuration:\n");
    printf("  Registers per thread           : %d\n", regsPerThread);
    printf("  Threads per block (max)        : %d\n", maxThreadsPerBlock);
    printf("  Total registers needed per block: %d\n", requiredRegsPerBlock);

    if (requiredRegsPerBlock <= prop.regsPerBlock) {
        printf("\nLaunch is possible: required registers (%d) <= max registers per block (%d).\n",
               requiredRegsPerBlock, prop.regsPerBlock);
    } else {
        printf("\nLaunch is NOT possible: required registers (%d) > max registers per block (%d).\n",
               requiredRegsPerBlock, prop.regsPerBlock);
        return -1;
    }

    // Launch the dummy kernel to confirm launch succeeds
    dim3 grid(1);
    dim3 block(maxThreadsPerBlock);
    dummyKernel<<<grid, block>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceSynchronize();
    printf("Kernel executed successfully.\n");

    return 0;
}
```