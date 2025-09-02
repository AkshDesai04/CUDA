/*
Pass an offset value to the kernel. Have each thread print blockIdx.x + offset.

[Thinking]
The task is to write a CUDA program that demonstrates passing an integer offset
value from the host to a kernel, and having each thread in the grid print the
expression `blockIdx.x + offset`.  The aim is simple: illustrate argument
passing, kernel launch configuration, and device side printing with `printf`.

Key points considered in the design:
1. **Kernel signature** – The kernel takes an `int` offset parameter.
2. **Printing** – CUDA supports `printf` on devices with compute capability
   2.0 or higher.  Each thread will output its thread index and the computed
   value for clarity.
3. **Grid launch** – To show that multiple blocks exist, the grid will be
   launched with several blocks (`gridDim.x = 4`).  Within each block, a few
   threads are launched (`blockDim.x = 2`) so that the kernel execution can
   be observed.
4. **Host code** – The host sets an offset value, copies it implicitly to
   the device when launching the kernel, then synchronizes to ensure all
   device prints complete before the program exits.
5. **Error checking** – A simple macro (`CUDA_CHECK`) is used to capture
   and report any CUDA runtime errors.

The resulting program will compile with `nvcc` and, when run, will print
lines like:

    [Block 0] Thread 0: blockIdx.x + offset = 0
    [Block 0] Thread 1: blockIdx.x + offset = 0
    [Block 1] Thread 0: blockIdx.x + offset = 1
    ...
    
This clearly demonstrates that the offset is successfully passed to the
kernel and used in the computation.

Now the full `.cu` file follows.  
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",    \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),        \
                    cudaGetErrorString(err));                             \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* Kernel that receives an offset and prints blockIdx.x + offset */
__global__ void offsetKernel(int offset) {
    /* Use thread index for more detailed output */
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    /* Print the computed value */
    printf("[Block %d] Thread %d: blockIdx.x + offset = %d\n", bid, tid, bid + offset);
}

int main(void) {
    /* Set the offset value */
    int offset = 5;

    /* Define grid and block dimensions */
    dim3 gridDim(4);          // 4 blocks
    dim3 blockDim(2);         // 2 threads per block

    /* Launch the kernel */
    offsetKernel<<<gridDim, blockDim>>>(offset);

    /* Wait for GPU to finish */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Reset device (optional) */
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
