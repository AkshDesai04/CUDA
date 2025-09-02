```cuda
/* 
   Aim of the program:
   Atomics on floating point numbers were not always supported. Check your device's compute capability. Modern GPUs support them for `float` and `double`.

   Thinking:
   1. The goal is to demonstrate how to query a CUDA-capable GPU’s compute capability
      and determine whether floating‑point atomic operations are available for
      single‑precision (`float`) and double‑precision (`double`) data types.
   2. CUDA provides `cudaGetDeviceProperties()` to obtain the device’s major and
      minor compute capability values.  Atomic operations on `float` were added
      in compute capability 3.5, and atomic operations on `double` were added
      in compute capability 6.0.
   3. A straightforward CUDA program can query these properties on the first
      device, then print whether atomic support is available for each type.
   4. No kernel launches are required; we only need host-side logic.  If one
      wished to test the atomic functions, a separate kernel would be needed
      and would have to be compiled for an architecture that supports the
      required operations, but that is beyond the scope of the current
      requirement.
   5. The program will be written in standard CUDA C++, include the necessary
      headers, and be fully self‑contained in a single .cu file.  It can be
      compiled with `nvcc` and executed on any machine with an NVIDIA GPU.
   6. The output will look something like:
        Device 0: "GeForce GTX 1080 Ti"
        Compute Capability: 6.1
        Atomic operations on float: supported
        Atomic operations on double: supported
   7. This satisfies the prompt’s requirement to “check your device’s compute
      capability” and to inform the user about modern GPU support for atomic
      operations on floating point numbers.
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
        printf("No CUDA-capable device found.\n");
        return 0;
    }

    // For simplicity, query the first device (device 0)
    int device = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Device %d: \"%s\"\n", device, prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    // Determine atomic support
    bool floatAtomicSupported = (prop.major > 3) || (prop.major == 3 && prop.minor >= 5);
    bool doubleAtomicSupported = (prop.major > 6) || (prop.major == 6 && prop.minor >= 0);

    printf("Atomic operations on float: %s\n",
           floatAtomicSupported ? "supported" : "not supported");
    printf("Atomic operations on double: %s\n",
           doubleAtomicSupported ? "supported" : "not supported");

    return 0;
}
```