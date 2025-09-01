/*
After printing the warp size, add a comment explaining what a warp is and why its size is important for performance (e.g., memory coalescing, SIMT execution).

Thinking:
The requirement is to produce a CUDA C program that prints the warp size of the device.  The warp size is a hardware property that indicates how many threads execute in lockstep on a single Streaming Multiprocessor (SM).  The program will query the device properties via the CUDA runtime API `cudaGetDeviceProperties` and output the `warpSize` field.  After the print statement, a comment will be inserted that explains what a warp is and why its size matters for performance, touching on memory coalescing and SIMT execution.  No additional text or output files should be produced.  The code will be a single .cu file that can be compiled with `nvcc` and run on a CUDA-capable GPU. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get CUDA device (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get CUDA device properties (error code %s)!\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Warp size on device %d (%s): %d\n", device, prop.name, prop.warpSize);

    /* 
    Explanation: 
    A warp is a group of 32 threads that execute the same instruction in lockstep on a CUDA Streaming Multiprocessor (SM). 
    The warp size determines how many threads are scheduled together and directly affects performance because:
    - Memory coalescing: When threads in a warp access contiguous memory addresses, the hardware can combine these accesses into a single memory transaction, maximizing memory bandwidth.
    - SIMT execution: All threads in a warp follow the same instruction path; divergent branches cause serialization, reducing throughput.
    Therefore, understanding and utilizing the warp size is crucial for writing efficient CUDA kernels. 
    */
    return 0;
}
