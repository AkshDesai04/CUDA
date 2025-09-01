/*
Query and print the total amount of shared memory available per multiprocessor on device 0 in kilobytes (KB).
Thought process:
- We need to query CUDA device properties for device 0.
- The field `sharedMemPerMultiprocessor` (in bytes) holds the amount of shared memory per SM.
- Convert bytes to kilobytes by dividing by 1024.
- Use `cudaGetDeviceProperties` to fetch the properties.
- Add basic error checking for CUDA API calls.
- Print the result in a readable format.
- Keep the code minimal and self-contained.
- Include necessary headers: <cuda_runtime.h> and <stdio.h>.
- The program should compile with `nvcc` and run on a system with a CUDA-capable GPU.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int device = 0;
    cudaDeviceProp prop;

    // Check if device 0 exists
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return -1;
    }
    if (deviceCount <= device) {
        fprintf(stderr, "Device 0 does not exist. Only %d device(s) available.\n", deviceCount);
        return -1;
    }

    // Get properties of device 0
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return -1;
    }

    size_t sharedMemBytes = prop.sharedMemPerMultiprocessor;
    size_t sharedMemKB = sharedMemBytes / 1024; // Convert to KB

    printf("Shared memory per multiprocessor on device %d: %zu KB\n", device, sharedMemKB);

    return 0;
}
