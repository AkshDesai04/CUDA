```cuda
/*
Aim: Query the device properties for `double` precision performance relative to single precision.

Thinking:
- The goal is to introspect CUDA-capable devices and report how double‑precision
  floating point performance compares with single‑precision.  
- CUDA exposes a property called `singleToDoubleRate` in `cudaDeviceProp`.  
  This field gives the ratio of the number of double‑precision operations that can
  be executed per clock cycle relative to the number of single‑precision operations.
  For example, a value of 0.5 means double precision is half as fast as single
  precision on that device.
- We will:
  1. Enumerate all CUDA devices using `cudaGetDeviceCount`.
  2. For each device obtain its properties with `cudaGetDeviceProperties`.
  3. Print a summary including:
     - Device name
     - Compute capability (major.minor)
     - Number of multiprocessors
     - Clock rate (kHz)
     - `singleToDoubleRate` (relative performance)
  4. Compute a rough theoretical single‑precision GFLOPS figure using the clock
     rate and an assumption of 2 FP32 ops per cycle per SM (common for many
     architectures).  Using `singleToDoubleRate` we derive a corresponding
     double‑precision GFLOPS estimate.  These figures are illustrative only
     and not architecture‑specific; the primary value is the relative rate.
- Error handling is added for CUDA API calls.  
- The program is written in C/C++ CUDA and can be compiled with `nvcc` to
  produce a .cu executable.

*/

#include <cstdio>
#include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0)
    {
        fprintf(stderr, "No CUDA devices found.\n");
        return 0;
    }

    printf("Detected %d CUDA device(s).\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to get properties for device %d: %s\n",
                    dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Multi-Processor Count: %d\n", prop.multiProcessorCount);
        printf("  Clock Rate: %d kHz\n", prop.clockRate);
        printf("  Single-Precision to Double-Precision rate: %.3f\n",
               prop.singleToDoubleRate);

        // Rough theoretical GFLOPs calculation
        // Assume 2 FP32 ops per cycle per SM (typical for many architectures)
        const float fp32OpsPerCyclePerSM = 2.0f;
        const float gHz = prop.clockRate * 1e-3f; // convert kHz to MHz, then to GHz
        float singleGflops = gHz * prop.multiProcessorCount * fp32OpsPerCyclePerSM;
        float doubleGflops = singleGflops * prop.singleToDoubleRate;

        printf("  Rough Single-Precision GFLOPs: %.2f\n", singleGflops);
        printf("  Rough Double-Precision GFLOPs: %.2f\n\n", doubleGflops);
    }

    return 0;
}
```