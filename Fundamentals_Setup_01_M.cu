```c
/* 
Aim of the program: 
Query and print the major and minor compute capability numbers for the primary CUDA device separately. 

Thought process:
- The primary CUDA device is device 0. 
- Use cudaGetDeviceProperties to retrieve properties of the device into a cudaDeviceProp struct. 
- The struct contains fields `major` and `minor` that represent the compute capability. 
- Check for errors from cudaGetDeviceProperties and handle them gracefully. 
- Print the major and minor values separately using printf. 
- Use standard C headers and the CUDA runtime header. 
- Compile with nvcc: nvcc -o compute_capability query_compute_capability.cu 
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;  // Primary CUDA device
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties (error %d: %s)\n",
                err, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Print major and minor compute capability separately
    printf("Major compute capability: %d\n", prop.major);
    printf("Minor compute capability: %d\n", prop.minor);

    return EXIT_SUCCESS;
}
```