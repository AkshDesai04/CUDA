```cuda
/* 
Aim of the program: 
Write a program to determine if the primary GPU is an integrated (sharing memory with the CPU) or a discrete device. Print the result.

Thinking process:
- The CUDA runtime API provides a structure cudaDeviceProp that contains properties of a device.
- One of the fields in cudaDeviceProp is 'integrated', which indicates whether the GPU shares memory with the CPU (integrated) or not (discrete).
- To determine the primary GPU, we query device 0 (the default device used by CUDA).
- Steps:
  1. Include necessary headers: <cuda_runtime.h> for CUDA API and <stdio.h> for I/O.
  2. Use cudaGetDeviceProperties(&prop, 0) to fetch properties of device 0.
  3. Check for errors from the API call; if any, print an error message and exit.
  4. Examine prop.integrated:
     - If prop.integrated == 1, print "Integrated GPU".
     - Else, print "Discrete GPU".
- The program will exit with status 0 on success, or 1 if an error occurs.

This approach relies on the 'integrated' field, which is supported in CUDA versions from 5.0 onward. The program is written in C with CUDA extensions and will compile to a .cu file.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void)
{
    cudaDeviceProp prop;
    int device = 0;  // Primary GPU

    // Get device properties
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n",
                device, cudaGetErrorString(err));
        return 1;
    }

    // Determine if the GPU is integrated or discrete
    if (prop.integrated)
    {
        printf("Integrated GPU\n");
    }
    else
    {
        printf("Discrete GPU\n");
    }

    return 0;
}
```