```cuda
/*
Query and print the total amount of shared memory available per multiprocessor on device 0 in kilobytes (KB).

My thinking:
- The CUDA runtime provides a struct cudaDeviceProp that contains many device properties, including the shared memory per multiprocessor in bytes.
- We first set the device to device 0, then retrieve its properties with cudaGetDeviceProperties().
- The field sharedMemPerMultiprocessor holds the amount in bytes; converting to kilobytes is simply dividing by 1024.
- We then print the value. For safety, all CUDA API calls are wrapped in a helper macro that checks for errors and reports them.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error: %s (err_num=%d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

int main(void)
{
    int device = 0;
    cudaDeviceProp prop;

    /* Set the active device to device 0 */
    CUDA_CHECK(cudaSetDevice(device));

    /* Get properties of the device */
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    /* Convert shared memory per multiprocessor from bytes to kilobytes */
    unsigned int shared_mem_kb = prop.sharedMemPerMultiprocessor / 1024;

    printf("Shared memory per multiprocessor on device %d: %u KB\n",
           device, shared_mem_kb);

    return 0;
}
```