```c
/*
Pass an offset value to the kernel. Have each thread print `blockIdx.x + offset`.

Thinking:
1. The program needs to demonstrate passing an integer offset from the host to a CUDA kernel.
2. Each thread will compute the sum of its block index (`blockIdx.x`) and the passed offset.
3. The requirement is to print this sum from each thread. Since device-side printing is
   supported via `printf`, we will use that in the kernel.
4. For clarity and to make the output meaningful, the kernel will also print the thread
   and block identifiers, but the essential output required is the sum `blockIdx.x + offset`.
5. The offset is passed by value, so we don't need any device memory copy for it.
6. We will launch a small grid (e.g., 4 blocks, 2 threads per block) so that the
   output is easily readable.
7. Error checking is added around the kernel launch to catch any launch failures.
8. The host code will use the CUDA runtime API, so we include <cuda_runtime.h> and <cstdio>.
9. After launching the kernel, we synchronize and exit.
10. The entire program is self‑contained in a single .cu file and can be compiled with
    `nvcc -o offset_program offset_program.cu`.
*/

#include <cuda_runtime.h>
#include <cstdio>

// Kernel that prints blockIdx.x + offset for each thread
__global__ void offsetKernel(int offset)
{
    // Each thread calculates the value
    int value = blockIdx.x + offset;
    // Print the value
    printf("Thread %d in Block %d: blockIdx.x + offset = %d\n",
           threadIdx.x, blockIdx.x, value);
}

int main(void)
{
    // Define offset value to pass to the kernel
    int offset = 5;  // Example offset; can be changed

    // Define grid and block dimensions
    dim3 gridDim(4);   // 4 blocks
    dim3 blockDim(2);  // 2 threads per block

    // Launch the kernel
    offsetKernel<<<gridDim, blockDim>>>(offset);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA device synchronize error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
```