```cuda
/*
Aim of the program:
Query and print the maximum memory pitch in bytes allowed for memory allocations.

Thinking:
The CUDA runtime provides a device attribute named CUDA_DEVICE_ATTRIBUTE_MAXIMUM_PITCH
which reports the maximum pitch (in bytes) that can be used for memory allocations
via functions such as cudaMallocPitch. To retrieve this value we need to:

1. Include the CUDA runtime header <cuda_runtime.h>.
2. Determine the number of CUDA-capable devices with cudaGetDeviceCount.
3. For each device (or simply device 0 for simplicity), call cudaDeviceGetAttribute
   with the attribute CUDA_DEVICE_ATTRIBUTE_MAXIMUM_PITCH.
4. Check for errors after each CUDA API call and print descriptive error messages
   using cudaGetErrorString if necessary.
5. Print the retrieved maximum pitch value to standard output.
6. Ensure the program compiles as a .cu file and runs on a CUDA-capable system.

We will handle a single device (device 0) for clarity, but the code can be
easily extended to loop over all devices. The program uses simple error
checking and reports the maximum pitch in bytes.*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    int device = 0; // Use the first device
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    int maxPitch = 0;
    err = cudaDeviceGetAttribute(&maxPitch, cudaDevAttrMaxPitch, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Maximum memory pitch allowed for allocations on device %d: %d bytes\n",
           device, maxPitch);

    return EXIT_SUCCESS;
}
```