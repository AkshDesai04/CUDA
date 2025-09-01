/*
Aim of the program: Launch an 8x8 block. Have only the thread at (0, 0) print a message.

Thought process:

1. The requirement is to create a simple CUDA program that demonstrates launching a single block of size 8x8 threads.
2. Only the thread with indices (0,0) should output a message. We can use the built-in `printf` function available in device code to print from within the kernel.
3. The kernel itself does not need to do any computation other than printing when the condition `threadIdx.x == 0 && threadIdx.y == 0` is satisfied.
4. Host code will launch the kernel with one block (`<<<1, dim3(8,8)>>`) and then call `cudaDeviceSynchronize()` to ensure that the device has finished executing and that all `printf` output is flushed to the host console.
5. Basic error checking after the kernel launch and after synchronization is included to catch any CUDA errors during compilation or execution.
6. The program includes the standard `<stdio.h>` header for the host side, and `cuda_runtime.h` is automatically included by the compiler when compiling with `nvcc`. No external memory allocation is required for this simple demonstration.

Compile with:
    nvcc -o launch_block launch_block.cu
Run with:
    ./launch_block
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void print_if_root()
{
    // Only thread (0,0) prints the message
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("Hello from thread (0,0) in block (0,0)!\n");
    }
}

int main()
{
    // Launch one block of 8x8 threads
    print_if_root<<<1, dim3(8, 8)>>>();
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Synchronize to ensure all device work is finished and prints are flushed
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA synchronization error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Optionally reset device (not strictly necessary for this simple program)
    cudaDeviceReset();

    return 0;
}
