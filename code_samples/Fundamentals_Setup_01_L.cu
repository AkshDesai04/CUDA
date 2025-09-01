/*
Aim: Query and print the maximum number of threads that can be active on a single multiprocessor for the primary device.

Thinking:
- The CUDA Runtime API provides `cudaGetDeviceProperties` which fills a `cudaDeviceProp` structure containing device-specific information.
- One of the fields in `cudaDeviceProp` is `maxThreadsPerMultiProcessor`, which represents the maximum number of active threads that can reside on a single Streaming Multiprocessor (SM) for that device.
- The “primary device” is typically device 0, unless the user explicitly changes the current device. To be safe, we first ensure that the current device is set to 0 using `cudaSetDevice(0)`. This will return an error if the device index is out of range.
- After setting the device, we retrieve the properties for that device and then print out the desired field.
- We include a simple error-checking macro `CUDA_CHECK` that wraps CUDA runtime API calls and aborts the program if an error occurs. This keeps the code clean and ensures we catch any unexpected failures early.
- The program is written in C/C++ (using `<cstdio>` for printing) and compiled with `nvcc`. It contains a single `main` function that performs the query and outputs the result to standard output.
- No additional files or libraries are required; the program compiles to a `.cu` file that can be run on any machine with CUDA drivers installed.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in file '%s' in line %d : %s (%d)\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err), err); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void) {
    int device = 0;  // primary device

    // Ensure device 0 exists and set it as current
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable device found.\n");
        return EXIT_FAILURE;
    }
    if (device >= deviceCount) {
        fprintf(stderr, "Requested device %d exceeds available device count %d.\n", device, deviceCount);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(device));

    // Retrieve device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // Print the maximum number of threads that can be active on a single SM
    printf("Maximum number of active threads per SM on device %d (%s): %d\n",
           device, prop.name, prop.maxThreadsPerMultiProcessor);

    return EXIT_SUCCESS;
}
