```cuda
/*
Aim: Declare `dim3 grid(4);` and `dim3 block(64);`. Launch a kernel and print `gridDim.x` and `blockDim.x` from within the kernel to verify they match.

Thinking:
- We want a minimal CUDA program that demonstrates the use of dim3 for grid and block dimensions.
- Declare grid and block with the specified sizes.
- Implement a simple kernel that prints the device's built‑in variables `gridDim.x` and `blockDim.x`.
- Use `printf` inside the kernel; this requires a compute capability of at least 2.0, which is fine for most modern GPUs.
- After launching the kernel, call `cudaDeviceSynchronize()` to ensure all printf output is flushed to the host.
- Include basic error checking for kernel launch and synchronization.
- The program is self‑contained in a single .cu file, with includes and a `main()` function.
- No additional libraries or external files are required.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints grid and block dimensions
__global__ void printDimsKernel() {
    // Print grid and block dimensions from within the kernel
    printf("gridDim.x = %d, blockDim.x = %d\n", gridDim.x, blockDim.x);
}

int main() {
    // Declare grid and block with specified dimensions
    dim3 grid(4);   // 4 blocks in the x dimension
    dim3 block(64); // 64 threads per block in the x dimension

    // Launch the kernel
    printDimsKernel<<<grid, block>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for the device to finish and flush printf output
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Optionally reset the device (clean up)
    cudaDeviceReset();

    return 0;
}
```