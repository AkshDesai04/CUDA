```cpp
/*
   Aim of the program: Query the device properties for `double` precision performance relative to single precision.

   Thinking:
   1. Use the CUDA Runtime API to detect the number of available devices.
   2. For each device, obtain the `cudaDeviceProp` structure via `cudaGetDeviceProperties`.
   3. The CUDA Runtime provides a device attribute `cudaDevAttrDoublePrecisionPerfRatio`
      that reports the ratio of double‑precision throughput to single‑precision throughput
      (expressed as a percentage). This attribute is available on compute capability 2.x
      and higher; older GPUs will return an error, which we handle gracefully.
   4. Print the device name, compute capability, number of multiprocessors, max threads
      per block, and the double‑precision performance ratio.
   5. Error handling: if any CUDA API call fails, print the error message and exit.
   6. The program is self‑contained and can be compiled with `nvcc` (e.g., `nvcc -o query_double_perf query_double_perf.cu`).
   7. No external dependencies beyond the CUDA Runtime are required.

   The code below follows these steps.
*/
#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error getting properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);

        int doublePerfRatio;
        err = cudaDeviceGetAttribute(&doublePerfRatio,
                                     cudaDevAttrDoublePrecisionPerfRatio,
                                     dev);
        if (err == cudaSuccess) {
            printf("  Double Precision Perf Ratio (%% of single): %d%%\n", doublePerfRatio);
        } else {
            printf("  Double Precision Perf Ratio: Not available (%s)\n", cudaGetErrorString(err));
        }

        printf("\n");
    }

    return 0;
}
```