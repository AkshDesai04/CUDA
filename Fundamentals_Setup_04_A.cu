/*
Aim: After printing the warp size, add a comment explaining what a warp is and why its size is important for performance (e.g., memory coalescing, SIMT execution).

Thinking process:
1. The program is a minimal CUDA example that queries and prints the warp size of the current GPU device.
2. Warp size is a compile‑time constant in CUDA device code, accessible via the built‑in variable `warpSize`.
3. We will launch a trivial kernel that runs on a single thread block with a single thread. That thread will print the warp size.
4. After the `printf` line, we will insert a comment that explains what a warp is and why its size matters for performance, addressing memory coalescing and SIMT execution as requested.
5. On the host side we set the device, launch the kernel, and synchronize. We also check for launch errors.
6. The program includes necessary headers (`<stdio.h>`, `<cuda_runtime.h>`) and uses `cudaDeviceSynchronize()` to ensure the kernel output is flushed before the program exits.
7. No additional supporting files or text are emitted; only the complete `.cu` source code is output.

The resulting `.cu` file can be compiled with:
    nvcc -arch=sm_20 warp_size_example.cu -o warp_size_example
and then run to see the warp size printed followed by the explanatory comment in the source.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printWarpSize()
{
    // Print the warp size of the current device
    printf("Warp size: %d\n", warpSize);
    // A warp is a group of 32 (or the device's warp size) threads that execute the same instruction simultaneously (SIMT execution).
    // The warp size is important for performance because:
    // 1. Memory coalescing: Threads within a warp access contiguous memory addresses, enabling a single memory transaction.
    // 2. Execution efficiency: Divergence within a warp causes serialized execution, so keeping warp size consistent helps maximize parallelism.
}

int main()
{
    cudaError_t err;

    // Set device to the first CUDA device
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Launch the kernel with one block and one thread
    printWarpSize<<<1, 1>>>();

    // Wait for the kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Clean up and exit
    return 0;
}
