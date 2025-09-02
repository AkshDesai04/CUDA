/*
Aim: Query and print the maximum number of threads that can be active on a single multiprocessor for the primary device.

Thinking:
1. The “primary device” in CUDA parlance refers to the device that is currently set for the calling context. We can retrieve its ID with cudaGetDevice().
2. CUDA device properties expose a field called `maxThreadsPerMultiProcessor`, which represents the maximum number of active warps (or threads) that can reside on a single streaming multiprocessor (SM) for the device.
3. We need to:
   - Include the necessary headers (`<cuda_runtime.h>` and `<stdio.h>`).
   - Use `cudaGetDevice()` to fetch the current device ID.
   - Call `cudaGetDeviceProperties()` to obtain the properties structure.
   - Extract and print `prop.maxThreadsPerMultiProcessor`.
4. For robustness, we'll add a simple error-checking macro that will print a message and exit on any CUDA API error.
5. The program will compile as a CUDA source file (`.cu`) and can be built with `nvcc`. It will output the desired maximum thread count to `stdout`.

The final code is a self‑contained CUDA program that performs the required query and displays the result.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

int main(void) {
    int device_id = 0;
    CHECK_CUDA(cudaGetDevice(&device_id));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

    printf("Maximum number of threads that can be active on a single multiprocessor for device %d (%s): %d\n",
           device_id, prop.name, prop.maxThreadsPerMultiProcessor);

    return 0;
}
