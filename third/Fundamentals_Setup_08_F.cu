/*
Query and print the number of asynchronous engines the device has (`asyncEngineCount`).

Thinking:
The program needs to query the CUDA device properties to obtain the asyncEngineCount field,
which indicates how many asynchronous engines (i.e., DMA engines for asynchronous copy) the device has.
We'll use cudaGetDeviceCount to determine how many CUDA devices are available,
then for each device retrieve its properties via cudaGetDeviceProperties.
Error handling will be performed using a helper macro that checks the return status of CUDA API calls.
The program will print the asyncEngineCount for each device in a human‑readable format.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: \"%s\"\n", dev, prop.name);
        printf("  Async Engine Count: %d\n", prop.asyncEngineCount);
        printf("\n");
    }

    return EXIT_SUCCESS;
}
