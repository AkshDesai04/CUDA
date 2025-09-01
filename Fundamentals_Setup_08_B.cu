/*
Aim: Write a program that gets the runtime version and prints its major and minor versions separately.

Thought process:
- Use CUDA Runtime API cudaRuntimeGetVersion to obtain the CUDA runtime version as an integer.
- The integer is encoded as major*1000 + minor*10 (e.g., 11020 for CUDA 11.20).
- Compute major and minor: major = runtime / 1000; minor = (runtime % 1000) / 10.
- Add error checking for the CUDA call.
- Print major and minor versions to console.
- Use standard C headers for printing (cstdio).
- Keep the code simple, no GPU kernel needed.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main(void) {
    int runtimeVersion = 0;
    cudaError_t err = cudaRuntimeGetVersion(&runtimeVersion);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA runtime version: %s\n", cudaGetErrorString(err));
        return -1;
    }

    int major = runtimeVersion / 1000;
    int minor = (runtimeVersion % 1000) / 10;

    printf("CUDA Runtime Version: %d.%d\n", major, minor);
    printf("Major version: %d\n", major);
    printf("Minor version: %d\n", minor);

    return 0;
}
