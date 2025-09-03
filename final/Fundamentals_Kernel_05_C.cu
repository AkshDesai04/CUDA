```cuda
/*
Aim of the program:
Calculate the total number of threads on the host (`gridDim.x * blockDim.x`) and pass it to the kernel. Inside the kernel, print the global ID and the total number of threads.

Thinking:
The task requires computing the total number of CUDA threads launched on the host side, which is simply the product of the number of blocks in the x-dimension (`gridDim.x`) and the number of threads per block in the x-dimension (`blockDim.x`). This total must be passed as an argument to the device kernel so that each thread can access it. Inside the kernel, each thread can compute its unique global identifier (global ID) using the common formula:
```
globalID = blockIdx.x * blockDim.x + threadIdx.x;
```
The kernel will then use `printf` to output both the global ID and the total number of threads. Since `printf` from device code requires proper compiler flags (e.g., `-arch=sm_20` or higher and the `-rdc=true` option if necessary), we assume a recent CUDA toolkit that supports device printf. We also include basic error checking on the kernel launch and device synchronization to ensure that any runtime errors are caught.

The program will:
1. Define a grid and block configuration (e.g., 10 blocks of 256 threads each).
2. Compute `totalThreads` on the host.
3. Launch the kernel with `totalThreads` as a parameter.
4. Synchronize and exit gracefully.

The kernel prints each thread’s global ID and the total thread count, providing clear output that can be verified by inspecting the console.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Kernel that prints the global thread ID and total number of threads
__global__ void printThreadInfo(int totalThreads)
{
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Global ID: %d, Total Threads: %d\n", globalID, totalThreads);
}

int main()
{
    // Define grid and block dimensions
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = 10;

    // Compute total number of threads on the host
    int totalThreads = threadsPerBlock * blocksPerGrid;

    // Launch the kernel
    printThreadInfo<<<blocksPerGrid, threadsPerBlock>>>(totalThreads);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Synchronize to ensure all prints complete before exiting
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```