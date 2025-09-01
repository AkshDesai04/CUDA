/*
 * Query and print the `concurrentKernels` property.
 *
 * Thinking:
 * 1. The program needs to query device properties via CUDA runtime API.
 * 2. Use cudaGetDeviceCount to determine how many devices are present.
 * 3. For each device, call cudaGetDeviceProperties to get a cudaDeviceProp struct.
 * 4. The cudaDeviceProp struct contains a field called concurrentKernels which indicates
 *    whether the device can run kernels concurrently.
 * 5. Print the device index and the value of concurrentKernels.
 * 6. Handle errors gracefully: wrap CUDA calls with a helper macro to check return codes
 *    and report any failure.
 * 7. The program is a simple .cu file that can be compiled with nvcc and executed on
 *    a system with CUDA installed. No other functionality or dependencies are required.
 * 8. The code is written in plain C (with CUDA extensions) and includes necessary
 *    headers such as <stdio.h> and <cuda_runtime.h>.
 * 9. The output will be a list of devices with their concurrentKernels status.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                                    \
        }                                                           \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  concurrentKernels: %s\n",
               prop.concurrentKernels ? "Supported" : "Not Supported");
    }

    return EXIT_SUCCESS;
}
