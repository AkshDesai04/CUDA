```cuda
/*
 * Aim of the program:
 * Query and print the major and minor compute capability numbers for the primary CUDA device separately.
 *
 * Entire thinking:
 * 1. The CUDA runtime provides `cudaGetDevice` to get the currently active device
 *    (the primary device when the program starts, unless changed explicitly).
 * 2. Once we have the device ID, we can query device properties using
 *    `cudaGetDeviceProperties`.  The `cudaDeviceProp` struct contains `major`
 *    and `minor` fields that represent the compute capability version.
 * 3. For safety, we check the return codes of CUDA API calls and report
 *    errors if they occur.  The program will exit with a non‑zero status
 *    in case of failure.
 * 4. The output should display the major and minor numbers on separate lines
 *    for clarity, following the requirement.
 *
 * Implementation details:
 * - Include <cuda_runtime.h> and <stdio.h>.
 * - Use `cudaGetDevice(&device)` to obtain the current device.
 * - Allocate a `cudaDeviceProp` struct and call `cudaGetDeviceProperties(&prop, device)`.
 * - Print `prop.major` and `prop.minor` using `printf`.
 * - Return 0 on success, non‑zero on failure.
 *
 * The program is self‑contained in a single .cu file and can be compiled
 * with `nvcc -o compute_capability compute_capability.cu`.
 */

#include <cuda_runtime.h>
#include <stdio.h>

int main(void)
{
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Compute Capability Major: %d\n", prop.major);
    printf("Compute Capability Minor: %d\n", prop.minor);

    return 0;
}
```