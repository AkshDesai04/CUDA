/*
Aim of the program (word for word):
Launch with a 1D block of 64 threads. Have each thread print its lane ID within its warp (threadIdx.x % 32).

My thinking:
To satisfy the requirement we need a simple CUDA kernel that runs with exactly 64 threads in a single block. 
Each thread will calculate its lane ID by taking the remainder of its threadIdx.x divided by 32, because a warp is
32 threads. The kernel will print the thread's global index and its lane ID using device-side printf, which is
available in CUDA kernels starting with compute capability 2.0 and above. In the host code we launch the kernel
with one block of 64 threads, then synchronize the device to ensure all prints complete before the program exits.
We also include minimal error checking by synchronizing after kernel launch. The program is written in C++ style
CUDA (using <cstdio> for printf) and compiled with a standard CUDA compiler (e.g. nvcc). No additional
supporting files are needed. The output will list each thread and its corresponding lane ID.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void print_lane_id()
{
    // Compute lane ID within the warp
    int lane_id = threadIdx.x % 32;
    // Print the thread index and its lane ID
    printf("Thread %d lane ID: %d\n", threadIdx.x, lane_id);
}

int main()
{
    // Define block and grid dimensions
    const int threadsPerBlock = 64;
    const int numberOfBlocks = 1;

    // Launch the kernel
    print_lane_id<<<numberOfBlocks, threadsPerBlock>>>();

    // Wait for the kernel to finish and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
