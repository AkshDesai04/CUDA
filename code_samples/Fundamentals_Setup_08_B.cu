/*
Aim of the program: Write a program that gets the runtime version and prints its major and minor versions separately.

Thinking:
- We need to write a CUDA C/C++ (.cu) file that queries the CUDA Runtime API for its version.
- The CUDA runtime API provides `cudaRuntimeGetVersion(int *runtimeVersion)` which returns an integer encoding the major and minor version (e.g., 10010 for CUDA 10.1, 9020 for 9.2, 12030 for 12.3).
- The major version can be obtained by integer division by 1000.
- The minor version is derived from the remainder: `(runtimeVersion % 1000) / 10`. This works for all reported CUDA versions.
- We should include error handling: if `cudaRuntimeGetVersion` fails, we print an error message using `cudaGetErrorString`.
- For output, `printf` is sufficient and keeps the code simple and purely C/C++.
- The program will compile with `nvcc` and can be run to display the runtime major and minor numbers.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int runtime_version;
    cudaError_t err = cudaRuntimeGetVersion(&runtime_version);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving CUDA runtime version: %s\n", cudaGetErrorString(err));
        return -1;
    }

    int major = runtime_version / 1000;
    int minor = (runtime_version % 1000) / 10;

    printf("CUDA Runtime Version: %d.%d\n", major, minor);
    return 0;
}
