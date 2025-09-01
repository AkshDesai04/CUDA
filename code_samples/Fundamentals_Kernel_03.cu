```cpp
/* 
Aim of the program: 
Write a kernel that is launched with a 1D block of 64 threads. Each thread should print its own thread index within the block, which is accessible via `threadIdx.x`.

Thinking:
1. The program is a CUDA C/C++ program that demonstrates device-side printing using `printf`.  
2. A kernel will be defined that prints the value of `threadIdx.x`.  
3. The kernel launch configuration should be a single block (`gridDim.x = 1`) with 64 threads in the block (`blockDim.x = 64`).  
4. After launching the kernel we must synchronize the device to ensure that all prints are flushed before the program exits.  
5. Basic error checking will be added to catch any CUDA API failures.  
6. Since `printf` is used from the device, the code should compile with a GPU architecture that supports it (compute capability ≥ 2.0).  
7. The entire code is self‑contained and can be compiled with `nvcc` to produce an executable that prints thread indices from 0 to 63, one per line. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the thread index within its block
__global__ void printThreadIdx()
{
    printf("Thread %d\n", threadIdx.x);
}

int main()
{
    // Launch the kernel with 1 block of 64 threads
    printThreadIdx<<<1, 64>>>();

    // Wait for the kernel to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset the device (optional)
    cudaDeviceReset();

    return 0;
}
```