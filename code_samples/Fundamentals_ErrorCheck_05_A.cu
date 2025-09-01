/*
Aim: First, get the device count. Then, try to set the device to `count` (which is an invalid index, since they are 0-based).

Thinking:
1. Include necessary headers: <cuda_runtime.h> for CUDA runtime API and <stdio.h> for I/O.
2. In main, call cudaGetDeviceCount(&count) to retrieve the number of available CUDA devices.
3. Print the device count for visibility.
4. Attempt to set the device using cudaSetDevice(count). Since valid device indices are 0 to count-1, using count should trigger an error (cudaErrorInvalidDevice).
5. Capture the return value of cudaSetDevice and print an error message if it fails, using cudaGetErrorString to obtain a human-readable description.
6. Optionally, call cudaGetLastError to clear any pending error (though not strictly necessary for this simple demo).
7. Exit with appropriate return code based on success or failure.
8. Compile with nvcc, e.g., nvcc -o invalid_device invalid_device.cu
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err;

    // Get the number of CUDA-capable devices
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    // Attempt to set the device to an invalid index (count)
    printf("Attempting to set device to index %d (invalid index)...\n", deviceCount);
    err = cudaSetDevice(deviceCount);
    if (err != cudaSuccess) {
        printf("cudaSetDevice returned error: %s\n", cudaGetErrorString(err));
    } else {
        printf("cudaSetDevice succeeded (unexpected).\n");
    }

    // Optionally, clear any error state
    cudaGetLastError();

    return 0;
}
