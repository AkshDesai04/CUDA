/*
 * Aim: Modify the `printf` to include the block's ID: `printf("Hello from block %d!\n", blockIdx.x);`. Launch with 8 blocks.
 *
 * Thinking:
 * The goal is to adjust the kernel's output so that each block reports its own block index in the printf message.
 * In CUDA, device-side printf can use the built-in variable blockIdx.x to obtain the current block's ID.
 * The kernel will therefore be very simple: it will just call printf with the format string that includes blockIdx.x.
 *
 * For execution we want 8 blocks, each with a single thread (1 thread per block is sufficient since we only need one
 * printf per block). Thus the launch configuration will be <<<8, 1>>>.
 *
 * We will also add basic error checking after the kernel launch to catch any launch failures, and call
 * cudaDeviceSynchronize() to ensure that the kernel completes before the host program exits and before we
 * attempt to read any device output.
 *
 * The code below follows the usual pattern for a minimal CUDA program: include <stdio.h> for printf,
 * <cuda_runtime.h> for the runtime API, declare the __global__ kernel, launch it in main, check for errors,
 * and synchronize. The kernel uses the modified printf statement requested.
 *
 * The resulting program can be compiled with nvcc, e.g.:
 *    nvcc -o hello hello.cu
 * and run with:
 *    ./hello
 * The output will show eight lines, each from a different block index.
 */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel()
{
    // Print a greeting from the current block
    printf("Hello from block %d!\n", blockIdx.x);
}

int main()
{
    // Launch the kernel with 8 blocks, 1 thread per block
    helloKernel<<<8, 1>>>();

    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for the device to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}
