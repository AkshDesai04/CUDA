```cuda
/*
 * Aim of the program:
 * Write a program that gets the runtime version and prints its major and minor versions separately.
 *
 * Thinking process:
 * 1. We need a CUDA program that queries the CUDA runtime API for its version.
 *    The function `cudaRuntimeGetVersion(int *runtimeVersion)` returns the
 *    combined version number (e.g., 11020 for CUDA 11.2). This value is an
 *    integer where the major version is the thousands digit, the minor version
 *    is the hundreds digit, and any sub‑minor version is the tens digit.
 *
 * 2. To extract the major and minor components:
 *    - major = runtimeVersion / 1000
 *    - minor = (runtimeVersion % 1000) / 10
 *
 * 3. We will:
 *    - Call `cudaRuntimeGetVersion`.
 *    - Check for errors using `cudaGetErrorString`.
 *    - Print the major and minor versions separately.
 *
 * 4. The program is a simple host‑only CUDA C program. It can be compiled
 *    with `nvcc` and will run on any system that has a CUDA toolkit installed.
 *
 * 5. Edge cases:
 *    - If `cudaRuntimeGetVersion` fails, we print an error and exit.
 *
 * 6. The output will be:
 *    "CUDA Runtime Version: <major>.<minor>"
 *    For example, if the runtime is 11.2, the output will be:
 *    "CUDA Runtime Version: 11.2"
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int runtime_version = 0;
    cudaError_t err = cudaRuntimeGetVersion(&runtime_version);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA runtime version: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    int major = runtime_version / 1000;
    int minor = (runtime_version % 1000) / 10;

    printf("CUDA Runtime Version: %d.%d\n", major, minor);

    return 0;
}
```