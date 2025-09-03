```cpp
/*
 * Write a program that gets the runtime version and prints its major and minor versions separately.
 *
 * Thinking:
 * - To get the CUDA runtime version, we use the CUDA Runtime API function `cudaRuntimeGetVersion()`.
 * - This function returns an integer that encodes the major and minor version of the CUDA runtime.
 *   The format is `major * 1000 + minor * 10`. For example, version 11.2 is represented as 11020.
 * - To extract the major and minor components:
 *   * major = version / 1000
 *   * minor = (version % 1000) / 10
 * - We'll check for errors from `cudaRuntimeGetVersion()` using the returned `cudaError_t`.
 *   If an error occurs, we print an error message and exit with a non-zero status.
 * - If successful, we print the major and minor versions separately to the console.
 * - The program is a minimal CUDA host-only program that can be compiled with `nvcc`.
 *
 * End of thinking.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main()
{
    int runtimeVersion = 0;
    cudaError_t err = cudaRuntimeGetVersion(&runtimeVersion);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA runtime version: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    int major = runtimeVersion / 1000;
    int minor = (runtimeVersion % 1000) / 10;

    printf("CUDA Runtime Version: %d.%d\n", major, minor);

    return EXIT_SUCCESS;
}
```