```cpp
/*
 * Aim: Use `nvcc --version` on the command line and compare its output to what `cudaRuntimeGetVersion` reports in your program.
 *
 * Thinking:
 * 1. The user is expected to run `nvcc --version` externally to get the compiler version (e.g., "Cuda compilation tools, release 11.8").
 * 2. In the CUDA program we can query the runtime version by calling `cudaRuntimeGetVersion`.
 * 3. The runtime version is an integer, e.g., 11020 for CUDA 11.20. We can convert it to the familiar major.minor format.
 * 4. The program will print the runtime version and the human‑readable string, so the user can visually compare it with the nvcc output.
 * 5. No kernel launch is necessary; this is a host‑only program.
 * 6. Include basic error checking for the CUDA API call.
 * 7. Compile with `nvcc my_program.cu` and run the resulting binary. Compare the printed runtime version with the `nvcc --version` output.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int runtimeVersion = 0;
    cudaError_t err = cudaRuntimeGetVersion(&runtimeVersion);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error calling cudaRuntimeGetVersion: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Convert to human-readable major.minor
    int major = runtimeVersion / 1000;
    int minor = (runtimeVersion % 1000) / 10;
    int patch = runtimeVersion % 10;

    printf("CUDA Runtime Version reported by cudaRuntimeGetVersion(): %d\n", runtimeVersion);
    printf("Human-readable format: %d.%d.%d\n", major, minor, patch);

    return EXIT_SUCCESS;
}
```