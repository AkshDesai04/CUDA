```cpp
/* 
Aim of the program: 
Calculate and print the theoretical floating-point operations per second (FLOPS) for single-precision on your GPU. 
A rough estimate is `ClockRate_GHz * MultiProcessorCount * CoresPerSM * 2`. 
You'll need to look up the cores per SM for your GPU architecture. 

Thinking:
- Use CUDA Runtime API to query device properties.
- The clock rate is provided in kHz; convert to GHz.
- Determine the number of cores per streaming multiprocessor (SM) based on the compute capability (major/minor).
  A common mapping for many recent architectures is:
      CC 7.0-7.2 : 64 cores per SM
      CC 7.5-8.0 : 128 cores per SM
      CC 8.6-8.9 : 128 cores per SM
  For older architectures, additional cases can be added if needed.
- Compute theoretical single‑precision FLOPS as:
      FLOPS = ClockRate_GHz * MultiProcessorCount * CoresPerSM * 2
  The factor 2 accounts for the ability to issue 2 FP32 operations per cycle per core on many GPUs.
- Output the GPU name, compute capability, clock rate, MP count, cores per SM, and the estimated GFLOPS.

Implementation details:
- The program queries the first GPU device (device 0). If no device is found, it reports an error.
- The cores-per-SM function uses a switch‑case on compute capability.
- The final FLOPS value is printed as a double in GFLOPS (divide by 1e9 to convert to gigaflops).
*/

#include <cstdio>
#include <cuda_runtime.h>
#include <string>

// Returns the number of CUDA cores per Streaming Multiprocessor for a given compute capability
int getCoresPerSM(int major, int minor) {
    switch (major) {
        case 2: // Fermi
            if (minor == 0) return 32;
            if (minor == 1) return 48;
            break;
        case 3: // Kepler
            return 192;
        case 5: // Maxwell
            return 128;
        case 6: // Pascal
            if (minor == 0) return 64;
            if (minor == 1) return 128;
            break;
        case 7: // Volta/ Turing
            if (minor == 0) return 64;
            if (minor == 2) return 64;
            if (minor == 5) return 64;
            break;
        case 8: // Ampere
            if (minor == 0) return 64;
            if (minor == 6) return 128;
            if (minor == 9) return 128;
            break;
        default:
            break;
    }
    // Fallback if unknown architecture
    return 64;
}

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        printf("No CUDA-capable device found.\n");
        return 1;
    }

    int device = 0; // Use the first device
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    double clockRateGHz = static_cast<double>(prop.clockRate) / 1e6; // convert kHz to GHz
    int mpCount = prop.multiProcessorCount;
    int coresPerSM = getCoresPerSM(prop.major, prop.minor);
    double gflops = clockRateGHz * mpCount * coresPerSM * 2.0; // theoretical GFLOPS

    printf("Device %d: %s\n", device, prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Clock Rate: %.3f GHz\n", clockRateGHz);
    printf("Multi-Processor Count: %d\n", mpCount);
    printf("Cores per SM: %d\n", coresPerSM);
    printf("Estimated Single-Precision GFLOPS: %.2f\n", gflops);

    return 0;
}
```