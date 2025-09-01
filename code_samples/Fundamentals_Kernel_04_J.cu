```cpp
/*
Pass an offset value to the kernel. Have each thread print `blockIdx.x + offset`.

The goal of this program is to demonstrate how to pass a scalar argument from the host
to a CUDA kernel and use that argument inside the kernel to compute and print a
value that depends on the block index. Each thread will print the expression
`blockIdx.x + offset`. This simple example highlights the following key points:

1. Declaring and initializing a host variable (offset) that we want to pass to the
   kernel.
2. Passing that variable as a kernel argument during launch.
3. Using `printf` from device code to output per-thread results.  Device `printf`
   requires a GPU compute capability of at least 2.0 and the appropriate compiler
   flags (`-arch=sm_20` or higher).
4. Synchronizing the device and checking for errors after kernel execution.

The kernel itself is trivial: it receives an `int offset` and then each thread
calculates `int result = blockIdx.x + offset;` and prints it.  To keep the
output manageable, the example launches a small grid (e.g., 4 blocks with 1 thread
per block), but the code can be adapted to larger configurations.

Overall, the program serves as a minimal illustration of passing data from the host
to the device and performing a per-thread computation that involves both a built‑in
CUDA variable (`blockIdx.x`) and a user‑supplied offset.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

// Kernel that prints blockIdx.x + offset for each thread
__global__ void offsetKernel(int offset) {
    // Each thread in this example has one thread per block.
    // Compute the sum and print it.
    int result = blockIdx.x + offset;
    printf("Block %d: blockIdx.x + offset = %d\n", blockIdx.x, result);
}

int main(void) {
    // Set the offset value
    int offset = 10;

    // Define grid and block dimensions
    dim3 blockDim(1);          // 1 thread per block
    dim3 gridDim(4);           // 4 blocks

    // Launch the kernel
    offsetKernel<<<gridDim, blockDim>>>(offset);

    // Synchronize to ensure kernel completion
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset the device (optional)
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
```