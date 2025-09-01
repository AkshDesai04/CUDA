```cuda
/*
Use `cudaGetDeviceProperties` and print the `driverVersion` field from the properties struct.

Thinking:
1. The goal is to query the CUDA driver version that is reported by the device properties
   structure. The `cudaDeviceProp` struct contains a field named `driverVersion` which
   holds the version number in the form 0xABC where ABC is the major/minor/patch
   combined into a single integer.  For example, 11020 means driver version 11.2.0.
2. The program will:
   - Include <cuda_runtime.h> to access CUDA runtime API.
   - Call `cudaGetDeviceCount` to verify that at least one CUDA device is present.
   - Pick device 0 (the default device) and call `cudaGetDeviceProperties`.
   - Print the driver version field.  For readability, we can format it as a decimal
     number (the raw integer) and also convert it to the human‑readable version
     string by extracting major, minor and patch components.
3. Error handling:
   - Wrap CUDA API calls with a helper function `checkCudaErrors` that checks the
     return status and prints an error message if something fails.
   - In case no devices are found, print an informative message and exit.
4. The program will be compiled as a single .cu file, so the only code needed is
   the main function and the helper.  No kernels are launched; this is purely a
   host‑side query program.
5. Note: The `driverVersion` field is only available from CUDA 6.0 onwards, so the
   program assumes a reasonably recent CUDA toolkit.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro to check CUDA errors */
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            return EXIT_FAILURE;                                          \
        }                                                                  \
    } while (0)

/* Convert raw driver version integer to human readable string */
void printDriverVersion(int rawVersion) {
    int major = rawVersion / 1000;
    int minor = (rawVersion % 1000) / 10;
    int patch = rawVersion % 10;
    printf("Driver version (raw): %d\n", rawVersion);
    printf("Driver version (readable): %d.%d.%d\n", major, minor, patch);
}

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    int device = 0; // Use the first device
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Device %d: %s\n", device, prop.name);
    printDriverVersion(prop.driverVersion);

    return EXIT_SUCCESS;
}
```