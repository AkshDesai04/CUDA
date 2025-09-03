```c
/*
 * Aim of the program:
 * Query the device properties for `double` precision performance relative to single precision.
 *
 * Thinking:
 * 1. The CUDA runtime provides `cudaGetDeviceProperties` to retrieve information
 *    such as compute capability, total global memory, clock rate, number of
 *    multiprocessors, etc.
 * 2. Double precision performance is not directly exposed by the CUDA API, but
 *    it depends heavily on the GPU's compute capability.  Historically, NVIDIA
 *    GPUs have different double‑to‑single precision throughput ratios:
 *      - Compute capability ≥ 7.0 (Turing, Ampere): 1.0 (double precision can be
 *        performed at the same rate as single precision on newer GPUs with
 *        hardware support).
 *      - Compute capability 6.x (Pascal): ~0.5 (half the throughput of single).
 *      - Compute capability 5.2 (Maxwell): ~0.5 (half the throughput).
 *      - Compute capability 5.0 (Kepler): ~0.25 (quarter throughput).
 *      - Compute capability 3.x (Kepler): ~0.5 (half throughput).
 *      - Compute capability < 3.0 (older GPUs): double precision is either
 *        unsupported or extremely slow (effectively 0.0 relative throughput).
 * 3. The program will:
 *      a. Get the number of devices.
 *      b. Choose device 0 (or the first device) for simplicity.
 *      c. Retrieve its properties with `cudaGetDeviceProperties`.
 *      d. Determine if double precision is supported (compute capability ≥ 3.0).
 *      e. Map the compute capability to a relative performance ratio using a
 *         helper function.
 *      f. Print the device name, compute capability, memory, clock rate,
 *         multiprocessor count, double precision support flag, and the
 *         relative performance ratio.
 *
 * The output will be human readable and will give a quick insight into how
 * double precision compares to single precision on the selected GPU.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Helper to map compute capability to double-to-single precision ratio
double getDoubleToSingleRatio(int major, int minor)
{
    // Build a single integer for comparison
    int cc = major * 10 + minor; // e.g., 7.5 -> 75

    if (cc >= 70) {            // 7.0 and above
        return 1.0;
    } else if (cc >= 60) {     // 6.0 to 6.9
        return 0.5;
    } else if (cc >= 52) {     // 5.2 to 5.9
        return 0.5;
    } else if (cc >= 50) {     // 5.0 to 5.1
        return 0.25;
    } else if (cc >= 35) {     // 3.5 to 5.0
        return 0.5;
    } else if (cc >= 30) {     // 3.0 to 3.4
        return 0.5;
    } else {
        return 0.0;            // Double precision unsupported or very slow
    }
}

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0)
    {
        fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }

    // Use the first device
    int dev = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d: %s\n", dev, prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total Global Memory: %.2f GB\n", (double)prop.totalGlobalMem / (1024 * 1024 * 1024));
    printf("  Clock Rate: %.2f MHz\n", prop.clockRate / 1000.0);
    printf("  Multiprocessor Count: %d\n", prop.multiProcessorCount);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("  Double Precision Support: %s\n",
           (prop.major >= 3) ? "Yes" : "No");

    double ratio = getDoubleToSingleRatio(prop.major, prop.minor);
    printf("  Approximate Double-to-Single Precision Ratio: %.2f\n", ratio);

    return 0;
}
```