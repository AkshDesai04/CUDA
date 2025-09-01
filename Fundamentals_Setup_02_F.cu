```cuda
/*
Write a program that finds the device with the highest compute capability and prints its name and index.
This comment contains my full reasoning for writing the code.

1. **Goal**: Enumerate all CUDA-capable devices available on the system, determine which one has the highest compute capability (major, then minor), and output that device's index and name.
2. **CUDA Runtime API**: Use `cudaGetDeviceCount` to find how many devices are present. For each device, retrieve its properties with `cudaGetDeviceProperties`, which provides fields `major`, `minor`, and `name`.
3. **Comparison Logic**:
   - The compute capability is a tuple `(major, minor)`. A higher `major` is always better; if majors are equal, compare `minor`.
   - Keep track of the best device's index and compute capability while iterating.
4. **Printing**: Once the best device is identified, use standard `printf` to display its index and name in a readable format.
5. **Error handling**:
   - If `cudaGetDeviceCount` returns an error or zero devices, print an appropriate message and exit.
   - Check each CUDA API call for success; if any fails, print an error and exit.
6. **Compilation**: The file is a `.cu` file and can be compiled with `nvcc`. No GPU kernels are required; only host code is used.
7. **Structure**:
   - `#include` the necessary headers.
   - `main` function performs the logic.
   - Return `0` on success, non-zero on failure.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    int bestIndex = -1;
    int bestMajor = -1;
    int bestMinor = -1;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", i, cudaGetErrorString(err));
            return 1;
        }

        // Compare compute capabilities
        if (prop.major > bestMajor || (prop.major == bestMajor && prop.minor > bestMinor)) {
            bestMajor = prop.major;
            bestMinor = prop.minor;
            bestIndex = i;
        }
    }

    if (bestIndex != -1) {
        cudaDeviceProp bestProp;
        err = cudaGetDeviceProperties(&bestProp, bestIndex);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for best device %d: %s\n", bestIndex, cudaGetErrorString(err));
            return 1;
        }
        printf("Device with highest compute capability: %s\n", bestProp.name);
        printf("Index: %d, Compute Capability: %d.%d\n", bestIndex, bestMajor, bestMinor);
    } else {
        printf("Could not determine the best device.\n");
    }

    return 0;
}
```