/*
Aim: Launch with a 1D block of 64 threads. Have each thread print its lane ID within its warp (threadIdx.x % 32).
This program demonstrates how to use a simple CUDA kernel to compute and output the lane ID of each thread
in a warp. In CUDA, a warp consists of 32 threads that execute in lockstep. The lane ID is simply the
thread index modulo 32. By launching 64 threads in a single block, we ensure that we have exactly two
full warps. Each thread will compute its lane ID and print it along with its global thread index.
The code includes basic error checking for kernel launch and device synchronization. It uses the
CUDA runtime API and standard C I/O functions. The kernel uses device printf, which is supported on
modern CUDA-capable GPUs. Running this program will output 64 lines, each indicating the thread
index and its corresponding lane ID within its warp.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void print_lane_id(void)
{
    int thread_id = threadIdx.x;
    int lane_id = thread_id % 32; // lane ID within the warp
    printf("Thread %d has lane ID %d\n", thread_id, lane_id);
}

int main(void)
{
    // Launch kernel with 1 block of 64 threads
    print_lane_id<<<1, 64>>>();
    CHECK_CUDA(cudaGetLastError());

    // Wait for the kernel to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset the device (optional but good practice)
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
