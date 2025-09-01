/*
Explain in a comment the concept of separate host and device memory address spaces.

My thinking:
- The goal is to produce a CUDA C source file (.cu) that includes a multiline comment explaining the fact that the host (CPU) and device (GPU) each have their own independent memory address spaces. 
- In CUDA, memory cannot be accessed directly across these spaces; data must be explicitly copied between them using functions such as cudaMemcpy.  
- The comment will describe why separate spaces exist (different hardware, architecture, caching, etc.) and the practical implications (explicit copy, separate allocation APIs).  
- To give the comment some context, the program will perform a trivial data transfer: allocate pinned host memory, allocate device memory, copy a value from host to device, launch a simple kernel that writes a result, copy back to host, and print it.  
- The code will be minimal, compile‑ready, and include basic error checking.  
- The entire program will be contained in a single .cu file with no extraneous output, as required by the developer instructions.  
*/

#include <cuda_runtime.h>
#include <stdio.h>

/* Error checking helper */
static inline void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* Simple kernel that writes a constant to device memory */
__global__ void write_value_kernel(int *d_data, int val)
{
    d_data[0] = val;
}

int main(void)
{
    const int N = 1;
    int h_value = 0;
    int *d_value;

    /* Allocate pinned (page‑locked) host memory for faster transfers */
    checkCudaError(cudaMallocHost((void **)&h_value, N * sizeof(int)), "cudaMallocHost");

    /* Allocate device memory */
    checkCudaError(cudaMalloc((void **)&d_value, N * sizeof(int)), "cudaMalloc");

    /* Copy data from host to device */
    checkCudaError(cudaMemcpy(d_value, h_value, N * sizeof(int), cudaMemcpyHostToDevice),
                   "cudaMemcpy HostToDevice");

    /* Launch kernel to write a value on the device */
    write_value_kernel<<<1, 1>>>(d_value, 42);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    /* Copy the result back to host */
    checkCudaError(cudaMemcpy(h_value, d_value, N * sizeof(int), cudaMemcpyDeviceToHost),
                   "cudaMemcpy DeviceToHost");

    /* Print the result */
    printf("Result from device: %d\n", h_value);

    /* Clean up */
    cudaFreeHost(h_value);
    cudaFree(d_value);

    return 0;
}
