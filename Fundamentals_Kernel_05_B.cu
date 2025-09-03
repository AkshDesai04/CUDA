```cpp
/* 
Aim of the program: Also print the thread's `blockIdx.x` and `threadIdx.x` next to its calculated global ID.

Thinking: 
1. The user wants a simple CUDA program that demonstrates how to compute and display the global thread ID along with the block and thread indices. 
2. Use a kernel that calls `printf` inside the GPU, which requires the device to support compute capability 2.0 or higher. 
3. Compute the global ID as `blockIdx.x * blockDim.x + threadIdx.x`. 
4. In the kernel, print the blockIdx.x, threadIdx.x, and the computed global ID. 
5. In `main()`, choose a modest grid and block size (e.g., 4 blocks with 8 threads each) for demonstration purposes. 
6. Launch the kernel, then synchronize and handle any errors. 
7. Add basic error checking after kernel launch and after `cudaDeviceSynchronize()`. 
8. Include necessary headers and use standard C++ style (though the kernel uses C printf). 
9. Ensure the code compiles with `nvcc` and produces output on the host terminal when executed. 
10. Wrap the entire code in a single .cu file as requested, with no additional files or text outside the comment block.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Kernel that prints blockIdx.x, threadIdx.x, and the global thread ID
__global__ void printThreadInfo()
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("BlockIdx.x=%d, ThreadIdx.x=%d, GlobalID=%d\n",
           blockIdx.x, threadIdx.x, global_id);
}

int main()
{
    // Define grid and block dimensions
    const int threadsPerBlock = 8;
    const int blocksPerGrid = 4;

    // Launch the kernel
    printThreadInfo<<<blocksPerGrid, threadsPerBlock>>>();
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for the device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Optional: reset the device
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
```