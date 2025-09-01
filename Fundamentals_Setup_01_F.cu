/*
Aim: Query and print the amount of shared memory available per thread block in bytes for the primary device.

Thinking:
- The program needs to query CUDA device properties for device 0 (the primary device).
- We use cudaGetDeviceProperties to obtain a cudaDeviceProp structure.
- The shared memory per block is stored in the field prop.sharedMemPerBlock.
- We print the value in bytes.
- Error handling: if cudaGetDeviceProperties fails, print an error message and exit with non-zero status.
- The program is written in C/C++ style with includes for <stdio.h> and <cuda_runtime.h>.
- Compile with nvcc: nvcc -o shared_mem_query shared_mem_query.cu
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0; // primary device
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d: sharedMemPerBlock = %u bytes\n", device, prop.sharedMemPerBlock);
    return 0;
}
