/*
Aim of the program: Create a simple Makefile to automate the compilation process.
This comment is included verbatim as requested.  
Below this comment I will write the CUDA program.  

Thought process:
1. The user requested a Makefile, but the developer instruction is higher priority and asks for a CUDA .cu file.  
2. Since no specific functionality was given, I'll provide a minimal CUDA program that demonstrates GPU execution: a kernel that prints a message.  
3. The program will consist of a __global__ kernel that writes to a device buffer and a host function that prints the buffer contents.  
4. The code will be self‑contained, compile with nvcc, and run on a device that supports printf from device.  
5. This simple program can be used as a template for more complex CUDA projects and can be built with a Makefile (which the user can create separately).  
6. The code will include error checking for CUDA API calls.  
7. The program will print "Hello from GPU!" when executed.  

This meets the developer requirement to produce a .cu file with a full program, while noting the user's original request in the comment.  
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel(char *msg)
{
    // Each thread writes its own message; we use only thread 0 for simplicity
    if (threadIdx.x == 0)
    {
        // Write a simple string to the device buffer
        const char *hello = "Hello from GPU!";
        for (int i = 0; hello[i] != '\0'; ++i)
        {
            msg[i] = hello[i];
        }
        msg[sizeof("Hello from GPU!") - 1] = '\0';
    }
}

int main(void)
{
    const int msgSize = 64;
    char *d_msg;
    char h_msg[msgSize];

    // Allocate device memory
    cudaError_t err = cudaMalloc((void **)&d_msg, msgSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Launch kernel
    helloKernel<<<1, 1>>>(d_msg);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_msg);
        return EXIT_FAILURE;
    }

    // Copy result back to host
    err = cudaMemcpy(h_msg, d_msg, msgSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_msg);
        return EXIT_FAILURE;
    }

    // Print message
    printf("%s\n", h_msg);

    // Clean up
    cudaFree(d_msg);
    return EXIT_SUCCESS;
}
