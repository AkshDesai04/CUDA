```cpp
/*
Aim: Write a kernel that is launched with a 1D grid of 8 blocks, where each block has only one thread. Each thread should print its own block index, which is accessible via `blockIdx.x`.

Thinking:
1. The requirement is very simple: we need a kernel that prints the block index.
2. Since each block should contain only one thread, the blockDim will be (1,1,1).
3. The grid dimension should be 8 blocks in the x dimension, so gridDim = (8,1,1).
4. Device printf is available in CUDA, so we can simply call printf inside the kernel.
5. To avoid any issues with printf buffering, we can synchronize after the kernel launch.
6. We'll use a simple main function that sets the device (optional), launches the kernel with the specified grid/block dims, synchronizes, and resets the device.
7. The code will include <stdio.h> for printf and <cuda_runtime.h> for CUDA runtime API functions.
8. Since we are using device printf, we must compile with `nvcc -arch=sm_XX` (the specific architecture will depend on the target GPU). For portability we can leave that to the user.
9. We will keep the code minimal and self‑contained in a single .cu file.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the block index
__global__ void printBlockIdx(void) {
    // Each block has a single thread, so threadIdx is always 0
    printf("Block index: %d\n", blockIdx.x);
}

int main(void) {
    // Launch configuration: 8 blocks, 1 thread per block
    dim3 gridDim(8);
    dim3 blockDim(1);

    // Launch the kernel
    printBlockIdx<<<gridDim, blockDim>>>();

    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA sync error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Optional: reset the device (clean up)
    cudaDeviceReset();

    return 0;
}
```