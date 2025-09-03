```cpp
/*
Aim of the program: Write a program that prints the CUDA driver version and the CUDA runtime version detected on your system. Use `cudaDriverGetVersion` and `cudaRuntimeGetVersion` and explain in a comment what the difference between the two is.

Thinking:
1. The program should be a self-contained CUDA C++ source file (.cu).
2. It needs to include the CUDA runtime header (`cuda_runtime.h`) to access both API calls.
3. The driver API function `cudaDriverGetVersion` returns the major and minor version numbers packed into an integer: 10010 for CUDA 10.1, etc.
4. The runtime API function `cudaRuntimeGetVersion` does the same but represents the version of the CUDA runtime library linked against the program.
5. The difference: The driver version is the actual NVIDIA driver installed on the system (which may support multiple CUDA versions). The runtime version is the CUDA Toolkit version used to compile/link the program; it may be older or newer than the driver. The driver must be at least as recent as the runtime to run the program.
6. The program will query both, check for success, and print the decoded versions (major.minor).
7. We will use standard output (`printf`) for simplicity.
8. Handle errors by printing the error string using `cudaGetErrorString`.
9. The main function will return 0 on success.
10. Ensure the code compiles with `nvcc` and runs on a system with CUDA installed.

*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main() {
    int driver_version = 0;
    int runtime_version = 0;

    // Get driver version
    cudaError_t drv_err = cudaDriverGetVersion(&driver_version);
    if (drv_err != cudaSuccess) {
        fprintf(stderr, "cudaDriverGetVersion failed: %s\n", cudaGetErrorString(drv_err));
        return EXIT_FAILURE;
    }

    // Get runtime version
    cudaError_t run_err = cudaRuntimeGetVersion(&runtime_version);
    if (run_err != cudaSuccess) {
        fprintf(stderr, "cudaRuntimeGetVersion failed: %s\n", cudaGetErrorString(run_err));
        return EXIT_FAILURE;
    }

    // Decode major and minor versions
    int drv_major = driver_version / 1000;
    int drv_minor = (driver_version % 1000) / 10;
    int run_major = runtime_version / 1000;
    int run_minor = (runtime_version % 1000) / 10;

    printf("CUDA Driver Version: %d.%d (encoded %d)\n", drv_major, drv_minor, driver_version);
    printf("CUDA Runtime Version: %d.%d (encoded %d)\n", run_major, run_minor, runtime_version);

    return EXIT_SUCCESS;
}
```