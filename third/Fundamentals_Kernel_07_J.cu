/*
Pass an integer offset `d`. Have threads print `(threadIdx.x + d, threadIdx.y + d)`.

**Thinking behind the solution:**
1. We need to accept an integer offset `d` from the user. For simplicity, the program reads it from the command line (`argv[1]`).
2. A CUDA kernel will be launched where each thread calculates its own indices `threadIdx.x` and `threadIdx.y` and adds `d` to them.
3. Printing from device code is done via `printf`, which requires a compute capability of at least 2.0. We use `__global__` and the kernel will call `printf` for each thread.
4. We choose a small block size (2x2) and a single block to keep the example minimal and easily observable.
5. After launching the kernel we call `cudaDeviceSynchronize()` to ensure all device prints are flushed before the host program exits.
6. Basic error checking is added for command-line parsing, kernel launch, and synchronization.
7. The code is written in a single `.cu` file and contains only the required includes, the kernel, the `main` function, and error handling.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void offsetKernel(int d)
{
    // Each thread prints its (threadIdx.x + d, threadIdx.y + d)
    printf("(%d, %d)\n", threadIdx.x + d, threadIdx.y + d);
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <offset_d>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int d = atoi(argv[1]);

    // Define a 2x2 block and a single block grid
    dim3 blockDim(2, 2);
    dim3 gridDim(1, 1);

    // Launch the kernel
    offsetKernel<<<gridDim, blockDim>>>(d);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Synchronize to ensure all printf output is flushed
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
