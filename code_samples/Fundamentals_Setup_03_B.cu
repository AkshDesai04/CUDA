```cuda
/*
Aim of the program:
Query and print the total amount of shared memory available per multiprocessor on device 0 in kilobytes (KB).

Thinking:
1. Use the CUDA runtime API to access device properties. The relevant struct is cudaDeviceProp.
2. The field shmemPerMultiprocessor gives the total amount of shared memory per SM in bytes.
3. To convert to kilobytes, divide by 1024.
4. Since we only care about device 0, we set the device to 0 using cudaSetDevice(0) and then retrieve properties via cudaGetDeviceProperties.
5. Perform minimal error checking to ensure the device is valid and the query succeeds.
6. Print the value in a human‑readable format.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            return EXIT_FAILURE;                                            \
        }                                                                   \
    } while (0)

int main(void) {
    // Set device to 0
    CUDA_CHECK(cudaSetDevice(0));

    // Retrieve properties for device 0
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // shmemPerMultiprocessor is in bytes; convert to kilobytes
    unsigned int sharedMemKB = prop.sharedMemPerMultiprocessor / 1024;

    printf("Total shared memory per multiprocessor on device 0: %u KB\n",
           sharedMemKB);

    return EXIT_SUCCESS;
}
```