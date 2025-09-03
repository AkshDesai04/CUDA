```cpp
/*
Aim of the program: Query and print the GPU's core clock rate in GHz.

Thinking:
1. The CUDA runtime API provides a structure `cudaDeviceProp` that contains device properties.
   Among them is `clockRate`, which gives the core clock in kHz.
2. To obtain this value, we need to:
   - Get the number of CUDA-capable devices with `cudaGetDeviceCount`.
   - For each device, retrieve its properties via `cudaGetDeviceProperties`.
3. Convert the kHz value to GHz by dividing by 1,000,000.0 (since 1 GHz = 1,000,000 kHz).
4. Print the result with a descriptive message.
5. Handle any potential CUDA errors in a simple manner (abort if an error occurs).
6. The program uses standard C/C++ (`printf`) and the CUDA runtime API; it compiles as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro to check CUDA errors
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",    \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),        \
                    cudaGetErrorString(err));                              \
            return EXIT_FAILURE;                                            \
        }                                                                   \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable device detected.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        // clockRate is in kHz; convert to GHz
        float coreClockGHz = prop.clockRate / 1e6f;

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Core clock rate: %.3f GHz\n", coreClockGHz);
    }

    return 0;
}
```