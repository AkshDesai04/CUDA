```cuda
/*
Extend the previous program. After copying data to the device, create a second host array of 256 integers. Copy the data from the device array back to this new host array using `cudaMemcpy` with the `cudaMemcpyDeviceToHost` kind. Verify that every element in the new host array matches the original.

Thinking:
1. Allocate an original host array of 256 integers and initialize it with values 0..255.
2. Allocate a device array of 256 integers using cudaMalloc.
3. Copy the original host array to the device array with cudaMemcpy using cudaMemcpyHostToDevice.
4. Allocate a second host array (newHost) of 256 integers.
5. Copy the device array back to newHost using cudaMemcpy with cudaMemcpyDeviceToHost.
6. Verify that each element in newHost matches the corresponding element in the original host array.
7. Report success or failure and free all allocated memory.
8. Include error checking for CUDA API calls via a helper macro.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main(void)
{
    const int N = 256;
    int hostOrig[N];
    int hostNew[N];

    // Initialize original host array
    for (int i = 0; i < N; ++i) {
        hostOrig[i] = i;
    }

    // Allocate device memory
    int *devPtr = NULL;
    CUDA_CHECK(cudaMalloc((void**)&devPtr, N * sizeof(int)));

    // Copy original host array to device
    CUDA_CHECK(cudaMemcpy(devPtr, hostOrig, N * sizeof(int), cudaMemcpyHostToDevice));

    // Copy data back from device to a new host array
    CUDA_CHECK(cudaMemcpy(hostNew, devPtr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify that all elements match
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (hostOrig[i] != hostNew[i]) {
            fprintf(stderr, "Mismatch at index %d: original=%d, new=%d\n",
                    i, hostOrig[i], hostNew[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Verification succeeded: all elements match.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Free device memory
    CUDA_CHECK(cudaFree(devPtr));

    return 0;
}
```