/*
Query and print the number of multiprocessors on the primary CUDA device (device 0).

Thinking:
To obtain the number of multiprocessors on the first CUDA device, we need to:
1. Include the CUDA runtime header (`cuda_runtime.h`) and the standard iostream.
2. Set device 0 as the current device using `cudaSetDevice(0)` (though this is optional for querying properties, it ensures we are working with the primary device).
3. Retrieve the device properties structure via `cudaGetDeviceProperties`.
4. The field `multiProcessorCount` in the returned structure holds the number of SMs.
5. Print this value to standard output.
6. Wrap CUDA API calls in a helper macro (`CUDA_CHECK`) that aborts on error for simplicity.
7. Return `0` from `main`.
*/

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << " at line "         \
                      << __LINE__ << ": " << cudaGetErrorString(err)         \
                      << std::endl;                                          \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main() {
    // Set device 0 as current device (optional but explicit)
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "Number of multiprocessors on device 0: "
              << prop.multiProcessorCount << std::endl;

    return 0;
}
