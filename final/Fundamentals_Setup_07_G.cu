```cpp
/*
Print the `major` and `minor` compute capability numbers for the primary device.
Thinking:
To satisfy the requirement, the program must query the compute capability
information of the primary CUDA device (device 0).  
The CUDA Runtime API provides cudaGetDeviceProperties() to obtain a
cudaDeviceProp structure that contains the fields `major` and `minor`.
The program should:
1. Include <cuda_runtime.h> for CUDA API declarations.
2. Call cudaGetDeviceProperties() for device 0.
3. Check for any CUDA errors.
4. Print the major and minor compute capability numbers in a readable
   format, e.g., "Compute Capability: X.Y".
5. Return 0 on success.
The program will compile with `nvcc` and run on any system with CUDA
installed and at least one CUDA device available.
*/

#include <cuda_runtime.h>
#include <cstdio>

int main(void) {
    // Assume device 0 is the primary device.
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n",
                device, cudaGetErrorString(err));
        return 1;
    }

    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    return 0;
}
```