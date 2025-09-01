```c
/*
If a kernel uses 64 registers per thread and the max threads per block is 1024, calculate the total registers required by a full block. Compare this to the max registers per block to see if such a launch is possible.

Thinking:
- We are asked to compute the total register usage for a full block given 64 registers per thread and 1024 threads.
- Total registers = 64 * 1024 = 65536.
- We need to compare this number to the hardware limit of registers per block.
- CUDA provides the attribute CUDA_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK (enumerated as cudaDevAttrMaxRegistersPerBlock) via cudaDeviceGetAttribute.
- We'll query that attribute for device 0 and print both numbers.
- We'll also print whether a launch with 1024 threads and 64 registers per thread would be feasible.
- The program does not need to launch any kernel; it simply performs the calculation and comparison.
- We include <cuda_runtime.h> and <stdio.h> and use the runtime API for simplicity.
- No other output or files are produced.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    const int regsPerThread = 64;
    const int threadsPerBlock = 1024;
    const int totalRegsNeeded = regsPerThread * threadsPerBlock;

    int maxRegsPerBlock = 0;
    cudaError_t err = cudaDeviceGetAttribute(&maxRegsPerBlock,
                                           cudaDevAttrMaxRegistersPerBlock,
                                           0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting max registers per block: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    printf("Registers per thread: %d\n", regsPerThread);
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Total registers needed for full block: %d\n", totalRegsNeeded);
    printf("Maximum registers per block on device 0: %d\n", maxRegsPerBlock);

    if (totalRegsNeeded <= maxRegsPerBlock) {
        printf("Launch is possible: Yes\n");
    } else {
        printf("Launch is possible: No\n");
    }

    return 0;
}
```